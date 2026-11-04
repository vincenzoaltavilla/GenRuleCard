from __future__ import annotations

import copy
import os
import warnings
from collections import Counter
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from joblib import Parallel, delayed, parallel_backend
from scipy.spatial.distance import cdist
from sklearn.metrics import fowlkes_mallows_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import trange

from ..core.artifacts import CorpusBundle, ProcessedDataset
from ..core.config import RuleCardConfig, VAEConfig
from ..rulecard_fit.scorecards import import_rulecard, scorecard_from_leaves, scorecard_outputs
from .model import (
    ScoreCardDataset,
    TreeScoreCardDataset,
    beta_schedule,
    build_vae,
    total_loss,
    tree_total_loss,
)
from .tree_matrix import build_tree_matrix_targets


def _bounded_abs_difference(values: np.ndarray) -> np.ndarray:
    values = np.abs(values)
    return values / (1.0 + values)


def _matrix_structural_distance(corpus: CorpusBundle) -> np.ndarray:
    rep = np.concatenate([corpus.scorecard_tensor.reshape(corpus.n_sc, -1), corpus.prior_target[:, None]], axis=1).astype(np.float32)
    distance = cdist(rep, rep).astype(np.float32)
    np.fill_diagonal(distance, 0.0)
    return distance


def _semantic_structural_distance(corpus: CorpusBundle) -> np.ndarray:
    pad = corpus.pad_target.astype(np.float32)
    active = pad < 0.5
    pattern = corpus.pat_target
    thresholds = corpus.thr_target.astype(np.float32)
    weights = corpus.weight_target.astype(np.float32)
    thr2_mask = corpus.thr2_mask.astype(np.float32) > 0.5
    prior = corpus.prior_target.astype(np.float32)

    distance = np.zeros((corpus.n_sc, corpus.n_sc), dtype=np.float32)
    for left in range(corpus.n_sc):
        for right in range(left + 1, corpus.n_sc):
            active_both = active[left] & active[right]
            same_pattern = active_both & (pattern[left] == pattern[right])

            padding_cost = float(np.abs(pad[left] - pad[right]).mean())
            pattern_cost = (
                float((pattern[left, active_both] != pattern[right, active_both]).mean())
                if active_both.any()
                else 0.0
            )

            threshold_cost = 0.0
            weight_cost = 0.0
            if same_pattern.any():
                thr1_diff = np.abs(thresholds[left, same_pattern, 0] - thresholds[right, same_pattern, 0])
                child_mask = thr2_mask[left, same_pattern] | thr2_mask[right, same_pattern]
                if child_mask.any():
                    thr2_diff = np.abs(
                        thresholds[left, same_pattern, 1][child_mask]
                        - thresholds[right, same_pattern, 1][child_mask]
                    )
                    threshold_cost = float(
                        (thr1_diff.sum() + thr2_diff.sum()) / (len(thr1_diff) + len(thr2_diff))
                    )
                else:
                    threshold_cost = float(thr1_diff.mean())

                weight_diff = weights[left, same_pattern] - weights[right, same_pattern]
                weight_cost = float(_bounded_abs_difference(weight_diff).mean())

            prior_cost = float(_bounded_abs_difference(np.asarray(prior[left] - prior[right])))
            scorecard_distance = (
                1.0 * padding_cost
                + 1.5 * pattern_cost
                + 0.8 * threshold_cost
                + 1.0 * weight_cost
                + 0.3 * prior_cost
            )
            distance[left, right] = scorecard_distance
            distance[right, left] = scorecard_distance
    return distance


def _row_activation_matrix(corpus: CorpusBundle, data: ProcessedDataset) -> np.ndarray:
    rows = corpus.tree_rows_df.copy()
    n_samples = data.X_train.shape[0]
    activations = np.zeros((corpus.n_sc, corpus.n_rows, n_samples), dtype=bool)
    card_to_pos = {identifier: position for position, identifier in enumerate(corpus.card_ids)}
    for row in rows.itertuples(index=False):
        if int(row.is_padding) == 1:
            continue
        card_pos = card_to_pos[int(row.scorecard_id)]
        row_pos = 2 * int(row.tree_idx) + int(row.row_in_tree)
        root_values = data.X_train[:, int(row.root_feat)]
        root_mask = root_values <= float(row.thr1) if int(row.row_in_tree) == 0 else root_values > float(row.thr1)
        if int(row.child_feat) >= 0:
            child_values = data.X_train[:, int(row.child_feat)]
            child_mask = child_values <= float(row.thr2)
            # The row represents a child split; use both child branches as the row support.
            child_mask = child_mask | (child_values > float(row.thr2))
            root_mask = root_mask & child_mask
        activations[card_pos, row_pos] = root_mask
    return activations


def _row_cost_matrix(
    left_card: int,
    left_rows: np.ndarray,
    right_card: int,
    right_rows: np.ndarray,
    pattern: np.ndarray,
    thresholds: np.ndarray,
    weights: np.ndarray,
    thr2_mask: np.ndarray,
    support_rate: np.ndarray,
) -> np.ndarray:
    left_pattern = pattern[left_card, left_rows][:, None]
    right_pattern = pattern[right_card, right_rows][None, :]
    same_pattern = left_pattern == right_pattern
    pattern_cost = (~same_pattern).astype(np.float32)

    support_cost = np.abs(
        support_rate[left_card, left_rows][:, None]
        - support_rate[right_card, right_rows][None, :]
    )

    thr1_cost = np.abs(
        thresholds[left_card, left_rows, 0][:, None]
        - thresholds[right_card, right_rows, 0][None, :]
    )
    child_mask = (
        thr2_mask[left_card, left_rows][:, None]
        | thr2_mask[right_card, right_rows][None, :]
    )
    thr2_cost = np.abs(
        thresholds[left_card, left_rows, 1][:, None]
        - thresholds[right_card, right_rows, 1][None, :]
    )
    threshold_cost = np.where(child_mask, 0.5 * (thr1_cost + thr2_cost), thr1_cost)
    threshold_cost = np.where(same_pattern, threshold_cost, 0.5)

    weight_diff = (
        weights[left_card, left_rows][:, None, :]
        - weights[right_card, right_rows][None, :, :]
    )
    weight_cost = _bounded_abs_difference(weight_diff).mean(axis=2)
    weight_cost = np.where(same_pattern, weight_cost, 0.5)

    return (
        1.4 * pattern_cost
        + 0.7 * threshold_cost
        + 0.9 * weight_cost
        + 0.8 * support_cost
    ).astype(np.float32)


def _chamfer_rule_set_cost(costs: np.ndarray, unmatched_cost: float) -> float:
    if costs.size == 0:
        return unmatched_cost
    n_left, n_right = costs.shape
    nearest_left = float(costs.min(axis=1).mean())
    nearest_right = float(costs.min(axis=0).mean())
    count_cost = unmatched_cost * abs(n_left - n_right) / max(n_left, n_right)
    return 0.5 * (nearest_left + nearest_right) + 0.25 * count_cost


def _semantic_matching_structural_distance(corpus: CorpusBundle, data: ProcessedDataset) -> np.ndarray:
    pad = corpus.pad_target.astype(np.float32)
    active = pad < 0.5
    pattern = corpus.pat_target
    thresholds = corpus.thr_target.astype(np.float32)
    weights = corpus.weight_target.astype(np.float32)
    thr2_mask = corpus.thr2_mask.astype(np.float32) > 0.5
    prior = corpus.prior_target.astype(np.float32)
    activations = _row_activation_matrix(corpus, data)
    support_rate = activations.mean(axis=2).astype(np.float32)

    distance = np.zeros((corpus.n_sc, corpus.n_sc), dtype=np.float32)
    for left in range(corpus.n_sc):
        for right in range(left + 1, corpus.n_sc):
            left_rows = np.flatnonzero(active[left])
            right_rows = np.flatnonzero(active[right])
            unmatched_cost = 2.0
            if len(left_rows) == 0 and len(right_rows) == 0:
                matching_cost = 0.0
            elif len(left_rows) == 0 or len(right_rows) == 0:
                matching_cost = unmatched_cost
            else:
                costs = _row_cost_matrix(
                    left, left_rows, right, right_rows,
                    pattern, thresholds, weights, thr2_mask, support_rate,
                )
                matching_cost = _chamfer_rule_set_cost(costs, unmatched_cost)
            prior_cost = float(_bounded_abs_difference(np.asarray(prior[left] - prior[right])))
            scorecard_distance = matching_cost + 0.3 * prior_cost
            distance[left, right] = scorecard_distance
            distance[right, left] = scorecard_distance
    return distance


def _partition_probe_indices(data: ProcessedDataset, max_size: int) -> np.ndarray:
    n_samples = len(data.y_train)
    if n_samples <= max_size:
        return np.arange(n_samples, dtype=np.int64)
    rng = np.random.default_rng(2026)
    classes, counts = np.unique(data.y_train, return_counts=True)
    allocations = np.maximum(1, np.floor(max_size * counts / counts.sum()).astype(int))
    while allocations.sum() > max_size:
        allocations[np.argmax(allocations)] -= 1
    while allocations.sum() < max_size:
        allocations[np.argmax(counts / counts.sum() - allocations / max_size)] += 1
    selected = []
    for label, size in zip(classes, allocations):
        idx = np.flatnonzero(data.y_train == label)
        selected.append(rng.choice(idx, size=int(size), replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def _conditions_mask(conditions: list[tuple], X: np.ndarray) -> np.ndarray:
    mask = np.ones(X.shape[0], dtype=bool)
    for feature, threshold, operator in conditions:
        values = X[:, int(feature)]
        threshold = float(threshold)
        if operator in ("<", "<="):
            mask &= values <= threshold
        elif operator in (">", ">="):
            mask &= values > threshold
        else:
            raise ValueError(f"Operatore non riconosciuto nella regola: {operator}")
    return mask


def _active_rule_partition_labels(corpus: CorpusBundle, data: ProcessedDataset, max_probe_size: int) -> np.ndarray:
    probe_idx = _partition_probe_indices(data, max_probe_size)
    X_probe = data.X_train[probe_idx]
    labels = np.zeros((corpus.n_sc, len(probe_idx)), dtype=np.int32)
    card_to_pos = {identifier: position for position, identifier in enumerate(corpus.card_ids)}
    grouped = corpus.corpus_df.sort_values(["scorecard_id"]).groupby("scorecard_id", sort=False)
    for identifier, leaves in grouped:
        card_pos = card_to_pos[int(identifier)]
        codes = np.zeros(len(probe_idx), dtype=object)
        for local_rule, row in enumerate(leaves.itertuples(index=False)):
            points = getattr(row, "points", None)
            if points is not None and int(points) == 0:
                continue
            mask = _conditions_mask(list(row.conditions_key), X_probe)
            bit = 1 << int(local_rule)
            codes[mask] = [int(value) | bit for value in codes[mask]]
        _, encoded = np.unique(codes, return_inverse=True)
        labels[card_pos] = encoded.astype(np.int32)
    return labels


def _partition_fmi_row(left: int, labels: np.ndarray) -> tuple[int, np.ndarray]:
    values = np.empty(labels.shape[0] - left - 1, dtype=np.float32)
    with warnings.catch_warnings():
        # Sono etichette di cluster valide: sklearn avvisa soltanto perché
        # alcune partizioni hanno molti cluster rispetto alle istanze.
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50%.*",
            category=UserWarning,
        )
        for offset, right in enumerate(range(left + 1, labels.shape[0])):
            values[offset] = 1.0 - float(
                fowlkes_mallows_score(labels[left], labels[right])
            )
    return left, values


def _partition_fmi_structural_distance(corpus: CorpusBundle, data: ProcessedDataset, config: VAEConfig) -> np.ndarray:
    labels = _active_rule_partition_labels(corpus, data, config.partition_probe_size)
    distance = np.zeros((corpus.n_sc, corpus.n_sc), dtype=np.float32)
    n_jobs = max(1, min(int(config.distance_n_jobs), corpus.n_sc - 1))
    started = time.perf_counter()
    if n_jobs == 1:
        rows = (_partition_fmi_row(left, labels) for left in range(corpus.n_sc - 1))
    else:
        # Processi separati: sklearn svolge parecchio lavoro Python per ogni
        # FMI, quindi i thread non aggirerebbero efficacemente il GIL.
        with parallel_backend("loky", inner_max_num_threads=1):
            rows = Parallel(n_jobs=n_jobs, batch_size=1, max_nbytes="1M")(
                delayed(_partition_fmi_row)(left, labels)
                for left in range(corpus.n_sc - 1)
            )
    for left, values in rows:
        distance[left, left + 1:] = values
        distance[left + 1:, left] = values
    print(
        f"D_FMI completata | scorecard={corpus.n_sc} | coppie="
        f"{corpus.n_sc * (corpus.n_sc - 1) // 2} | worker={n_jobs} | "
        f"secondi={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return distance


def _structural_distance(
    corpus: CorpusBundle,
    config: VAEConfig | None = None,
    data: ProcessedDataset | None = None,
) -> np.ndarray:
    mode = "matrix" if config is None else config.structural_distance_mode
    if mode == "matrix":
        return _matrix_structural_distance(corpus)
    if mode == "semantic":
        return _semantic_structural_distance(corpus)
    if mode == "semantic_matching":
        if data is None:
            raise ValueError("structural_distance_mode='semantic_matching' richiede il dataset processato.")
        return _semantic_matching_structural_distance(corpus, data)
    if mode == "partition_fmi":
        if data is None or config is None:
            raise ValueError("structural_distance_mode='partition_fmi' richiede config e dataset processato.")
        return _partition_fmi_structural_distance(corpus, data, config)
    raise ValueError(f"structural_distance_mode non supportato: {mode}")


def _training_uses_metric_distance(config: VAEConfig) -> bool:
    weights = config.loss_weights
    return any(float(value) > 0.0 for value in (
        weights.get("metric", 0.0),
        weights.get("local_metric", 0.0),
        config.tree_monitor_metric_weight,
        config.tree_monitor_local_metric_weight,
    ))


def _training_structural_distance(
    corpus: CorpusBundle,
    config: VAEConfig,
    data: ProcessedDataset | None,
) -> np.ndarray:
    if not _training_uses_metric_distance(config):
        return np.zeros((corpus.n_sc, corpus.n_sc), dtype=np.float32)
    return _structural_distance(corpus, config, data)


def _positive_probabilities(scorecard, X: np.ndarray) -> np.ndarray:
    probabilities, _ = scorecard_outputs(scorecard, X)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim == 2:
        return probabilities[:, -1]
    raise ValueError(f"Output inatteso da ScoreCard.predict_proba: {probabilities.shape}")


def _predictive_distance(
    corpus: CorpusBundle,
    data: ProcessedDataset,
    rulecard_config: RuleCardConfig,
    source_dir: Path | str | None,
) -> np.ndarray:
    _, _, scorecard_cls = import_rulecard(source_dir)
    cards_by_id = corpus.cards_df.set_index("scorecard_id")
    probabilities = []
    for identifier in corpus.card_ids:
        leaves = corpus.corpus_df.loc[corpus.corpus_df["scorecard_id"] == identifier].copy()
        prior = float(cards_by_id.loc[identifier, "prior_log_odds"])
        scorecard = scorecard_from_leaves(leaves, prior, data.model_classes, rulecard_config, scorecard_cls)
        probabilities.append(_positive_probabilities(scorecard, data.X_train))
    predictive_probabilities = np.vstack(probabilities).astype(np.float32)
    distance = (cdist(predictive_probabilities, predictive_probabilities) / np.sqrt(predictive_probabilities.shape[1])).astype(np.float32)
    np.fill_diagonal(distance, 0.0)
    return distance


def _minmax_distance(distance: np.ndarray) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float32)
    lower = float(distance.min())
    upper = float(distance.max())
    normalized = (distance - lower) / (upper - lower + 1e-8)
    np.fill_diagonal(normalized, 0.0)
    return normalized.astype(np.float32)


def _distance_stats(name: str, distance: np.ndarray) -> str:
    mask = ~np.eye(distance.shape[0], dtype=bool)
    values = distance[mask]
    return (
        f"{name}: min={values.min():.6f} | max={values.max():.6f} | "
        f"mean={values.mean():.6f} | std={values.std():.6f}"
    )


def _metric_target_distance(
    corpus: CorpusBundle,
    config: VAEConfig,
    structural_distance: np.ndarray,
    data: ProcessedDataset | None,
    rulecard_config: RuleCardConfig | None,
    source_dir: Path | str | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    target_name = config.metric_target.lower()
    predictive_distance = None
    if target_name in {"predictive", "mixed"}:
        if data is None or rulecard_config is None:
            raise ValueError("metric_target='predictive' o 'mixed' richiede data e rulecard_config.")
        predictive_distance = _predictive_distance(corpus, data, rulecard_config, source_dir)
    if target_name == "structural":
        target_distance = structural_distance
    elif target_name == "predictive":
        assert predictive_distance is not None
        target_distance = predictive_distance
    elif target_name == "mixed":
        assert predictive_distance is not None
        structural_norm = _minmax_distance(structural_distance)
        predictive_norm = _minmax_distance(predictive_distance)
        target_distance = config.metric_alpha * structural_norm + (1.0 - config.metric_alpha) * predictive_norm
        np.fill_diagonal(target_distance, 0.0)
    else:
        raise ValueError(f"metric_target non supportato: {config.metric_target}")
    return target_distance.astype(np.float32), predictive_distance


def _make_optimizer(model: torch.nn.Module, config: VAEConfig):
    return torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )


def _load_fine_tune_checkpoint(
    model: torch.nn.Module,
    config: VAEConfig,
    device: torch.device,
) -> Path | None:
    """Carica il checkpoint iniziale esplicito, se configurato."""
    configured = config.initial_checkpoint
    if configured is None:
        return None
    path = Path(configured).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint VAE per fine-tuning non trovato: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(
        "VAE fine-tuning | "
        f"checkpoint={path} | learning_rate={config.learning_rate:g}",
        flush=True,
    )
    return path


def _tree_monitored_value(metrics: dict[str, float], config: VAEConfig) -> float:
    if config.early_stopping_monitor == "total":
        return float(metrics["total"])
    if config.early_stopping_monitor == "metric":
        return float(metrics["metric"])
    if config.vae_profile == "fmi_f1_sig":
        return float(
            metrics["root"]
            + 0.5 * (metrics["left_child"] + metrics["right_child"])
            + metrics.get("signature_count", 0.0)
            + metrics.get("count_leaves", 0.0)
            + 0.5 * metrics.get("f1_loss", 0.0)
            + 0.5 * metrics.get("metric", 0.0)
        )
    return float(
        metrics["reconstruction"]
        + config.tree_monitor_metric_weight * metrics.get("metric", 0.0)
        + config.tree_monitor_local_metric_weight * metrics.get("local_metric", 0.0)
    )


def _decoded_predictive_rmse(
    model: torch.nn.Module,
    corpus: CorpusBundle,
    card_positions: np.ndarray,
    data: ProcessedDataset,
    rulecard_config: RuleCardConfig,
    source_dir: Path | str | None,
    vae_config: VAEConfig | None = None,
) -> dict[str, float]:
    """Decoded fidelity on train; validation and test are never accessed."""
    from .decoder import ConstrainedDecoder

    decoder = ConstrainedDecoder(model, corpus, data, rulecard_config, source_dir, vae_config)
    _, _, scorecard_cls = import_rulecard(source_dir)
    cards_by_id = corpus.cards_df.set_index("scorecard_id")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rmse_values: list[float] = []
    balanced_accuracy_values: list[float] = []
    macro_f1_values: list[float] = []
    positive_recall_values: list[float] = []
    with torch.no_grad():
        for position in card_positions:
            position = int(position)
            identifier = int(corpus.card_ids[position])
            matrix = torch.tensor(
                corpus.scorecard_tensor[position][None], dtype=torch.float32, device=device
            )
            prior_target = torch.tensor(
                [corpus.prior_target[position]], dtype=torch.float32, device=device
            )
            mu, _ = model.encode(matrix, prior_target)
            decoded = decoder.decode(mu[0].cpu().numpy(), scorecard_id=-1000 - position)
            leaves = corpus.corpus_df.loc[
                corpus.corpus_df["scorecard_id"] == identifier
            ].copy()
            prior = float(cards_by_id.loc[identifier, "prior_log_odds"])
            original = scorecard_from_leaves(
                leaves, prior, data.model_classes, rulecard_config, scorecard_cls
            )
            original_probability = _positive_probabilities(original, data.X_train)
            decoded_probability = _positive_probabilities(decoded.scorecard, data.X_train)
            rmse_values.append(float(np.sqrt(np.mean((decoded_probability - original_probability) ** 2))))
            original_prediction = original_probability >= 0.5
            decoded_prediction = decoded_probability >= 0.5
            recalls = []
            f1_scores = []
            for label in (False, True):
                truth = original_prediction == label
                predicted = decoded_prediction == label
                tp = float(np.logical_and(truth, predicted).sum())
                fn = float(np.logical_and(truth, ~predicted).sum())
                fp = float(np.logical_and(~truth, predicted).sum())
                recalls.append(0.0 if tp + fn == 0 else tp / (tp + fn))
                f1_scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            balanced_accuracy_values.append(float(np.mean(recalls)))
            macro_f1_values.append(float(np.mean(f1_scores)))
            positive_recall_values.append(float(recalls[1]))
    model.train(was_training)
    balanced_accuracy = float(np.mean(balanced_accuracy_values))
    macro_f1 = float(np.mean(macro_f1_values))
    positive_recall = float(np.mean(positive_recall_values))
    fidelity = 0.4 * balanced_accuracy + 0.4 * macro_f1 + 0.2 * positive_recall
    return {
        "rmse": float(np.mean(rmse_values)),
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "positive_recall": positive_recall,
        "fidelity": float(fidelity),
    }


def _run_epoch(model, loader, optimizer, training, epoch, config, device, pattern_weight, target_distance):
    model.train(training)
    totals = Counter()
    n = 0
    for batch in loader:
        card_idx, matrix, prior, pattern, threshold, weight, padding, m1, m2, mw = [
            item.to(device) for item in batch
        ]
        with torch.set_grad_enabled(training):
            output, mu, logvar = model(matrix, prior, sample=training)
            loss, components = total_loss(
                output, mu, logvar, card_idx, pattern, threshold, weight, padding, m1, m2, mw,
                prior, beta_schedule(epoch, config), pattern_weight, target_distance, config,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_n = len(matrix)
        n += batch_n
        for key, value in components.items():
            totals[key] += float(value.detach().cpu()) * batch_n
    return {key: value / n for key, value in totals.items()}


def _run_tree_epoch(
    model, loader, optimizer, training, epoch, config, device, target_distance,
    X_validation=None, y_validation=None,
):
    model.train(training)
    totals = Counter()
    n = 0
    for batch in loader:
        matrix = batch["x"].to(device)
        prior = batch["prior"].to(device)
        with torch.set_grad_enabled(training):
            output, mu, logvar = model(matrix, prior, sample=training)
            loss, components = tree_total_loss(
                output, batch, mu, logvar, beta_schedule(epoch, config),
                target_distance, config, X_validation, y_validation,
            )
            if training:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_n = len(matrix)
        n += batch_n
        for key, value in components.items():
            totals[key] += float(value.detach().cpu()) * batch_n
    return {key: value / n for key, value in totals.items()}


def _train_tree_vae(
    corpus: CorpusBundle,
    config: VAEConfig,
    seed: int,
    output_dir: Path,
    data: ProcessedDataset,
    rulecard_config: RuleCardConfig | None = None,
    source_dir: Path | str | None = None,
) -> tuple[torch.nn.Module, int, pd.DataFrame, pd.DataFrame, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        "VAE | "
        f"representation=tree | device={device.type} | latent_dim={config.latent_dim} | "
        f"hidden_dim={config.hidden_dim} | layers={config.n_layers} | heads={config.n_heads}"
    )
    targets = build_tree_matrix_targets(corpus, data)
    values, counts = np.unique(targets.tree_type, return_counts=True)
    type_counts = ", ".join(f"{int(value)}:{int(count)}" for value, count in zip(values, counts))
    print(
        "Tree matrix | "
        f"shape={targets.tree_tensor.shape} | input_dim={targets.tree_input_dim} | "
        f"type_counts={type_counts}"
    )

    all_idx = np.arange(corpus.n_sc)
    original_n = config.fine_tune_original_corpus_size
    if config.initial_checkpoint is not None and original_n is not None:
        if original_n > corpus.n_sc:
            raise ValueError(
                "fine_tune_original_corpus_size supera la dimensione del corpus."
            )
        original_idx = np.arange(original_n)
        original_train_idx, holdout_idx = train_test_split(
            original_idx, test_size=0.20, random_state=seed
        )
        new_idx = np.arange(original_n, corpus.n_sc)
        train_idx = np.concatenate([original_train_idx, new_idx])
        print(
            "VAE fine-tuning split | "
            f"train_originali={len(original_train_idx)} | "
            f"train_nuove={len(new_idx)} | holdout_originale={len(holdout_idx)}",
            flush=True,
        )
    else:
        train_idx, holdout_idx = train_test_split(
            all_idx, test_size=0.20, random_state=seed
        )
    train_loader = DataLoader(TreeScoreCardDataset(targets, train_idx), batch_size=config.batch_size, shuffle=True)
    holdout_loader = DataLoader(TreeScoreCardDataset(targets, holdout_idx), batch_size=config.batch_size, shuffle=False)
    full_loader = DataLoader(TreeScoreCardDataset(targets, all_idx), batch_size=config.batch_size, shuffle=True)

    structural_np = _training_structural_distance(corpus, config, data)
    target_np, predictive_np = _metric_target_distance(
        corpus, config, structural_np, data, rulecard_config, source_dir
    )
    target_distance = torch.tensor(target_np, dtype=torch.float32, device=device)
    X_validation = torch.tensor(data.X_validation, dtype=torch.float32, device=device)
    y_validation = torch.tensor(
        (data.y_validation == data.model_classes[-1]).astype(np.float32),
        dtype=torch.float32, device=device,
    )

    torch.manual_seed(seed)
    model = build_vae(corpus, config, device, data)
    fine_tune_checkpoint = _load_fine_tune_checkpoint(model, config, device)
    optimizer = _make_optimizer(model, config)
    history = []
    best_val, best_epoch, best_state, no_improvement = np.inf, 0, None, 0
    if fine_tune_checkpoint is not None:
        with torch.no_grad():
            initial_val_metrics = _run_tree_epoch(
                model, holdout_loader, None, False, 1, config, device,
                target_distance, X_validation, y_validation,
            )
        best_val = _tree_monitored_value(initial_val_metrics, config)
        best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": 0,
            "beta": beta_schedule(1, config),
            "lr": optimizer.param_groups[0]["lr"],
            "vae_earlystop_monitored": best_val,
            "fine_tune_initial_checkpoint": True,
            **{
                f"vae_train_{key}": np.nan
                for key in initial_val_metrics
            },
            **{
                f"vae_earlystop_{key}": value
                for key, value in initial_val_metrics.items()
            },
        })
        print(
            "VAE fine-tuning baseline | "
            f"epoch=0 | holdout_monitor={best_val:.6f}",
            flush=True,
        )
    started = time.time()
    progress = trange(
        1, config.epochs + 1, desc="TreeVAE early stopping", unit="epoch", leave=True,
        dynamic_ncols=True,
    )
    for epoch in progress:
        train_metrics = _run_tree_epoch(
            model, train_loader, optimizer, True, epoch, config, device, target_distance,
            X_validation, y_validation,
        )
        with torch.no_grad():
            val_metrics = _run_tree_epoch(
                model, holdout_loader, None, False, epoch, config, device, target_distance,
                X_validation, y_validation,
            )
        monitored_value = _tree_monitored_value(val_metrics, config)
        row = {
            "epoch": epoch,
            "beta": beta_schedule(epoch, config),
            "lr": optimizer.param_groups[0]["lr"],
            "vae_earlystop_monitored": monitored_value,
            "fine_tune_initial_checkpoint": False,
        }
        row.update({f"vae_train_{key}": value for key, value in train_metrics.items()})
        row.update({f"vae_earlystop_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        progress.set_postfix(
            train=f"{train_metrics['total']:.4f}",
            val=f"{val_metrics['total']:.4f}",
            best=f"{best_val:.4f}" if np.isfinite(best_val) else "inf",
        )
        if monitored_value < best_val - config.min_delta:
            best_val, best_epoch, best_state, no_improvement = (
                monitored_value,
                epoch,
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            no_improvement += 1
        if no_improvement >= config.patience:
            print(
                f"TreeVAE early stopping: epoch={epoch}, best_epoch={best_epoch}, "
                f"best_monitor={best_val:.6f}"
            )
            break
    progress.close()
    if best_state is None:
        raise RuntimeError("Il TreeVAE non ha prodotto uno stato valido.")
    early_history = pd.DataFrame(history)
    early_history.to_csv(output_dir / "vae_earlystop_history.csv", index=False)
    torch.save(best_state, output_dir / "vae_earlystop_state.pt")

    if config.deployment_checkpoint == "earlystop":
        model.load_state_dict(best_state)
        model.eval()
        final_history = pd.DataFrame(columns=["epoch", "full_T_total"])
        final_history.to_csv(output_dir / "vae_final_train_history.csv", index=False)
        if os.environ.get("GENRULECARD_CLEAN_LOGS") == "1":
            print(
                "TreeVAE 80/20 selezionato per diagnostiche e GA | "
                f"best_epoch={best_epoch}"
            )
        else:
            print(
                f"TreeVAE 80/20 selezionato per diagnostiche e GA | "
                f"best_epoch={best_epoch} | refit full T saltato | {time.time() - started:.1f}s"
            )
        if not _training_uses_metric_distance(config):
            structural_np = _structural_distance(corpus, config, data)
        return model, best_epoch, early_history, final_history, structural_np

    torch.manual_seed(seed)
    model = build_vae(corpus, config, device, data)
    optimizer = _make_optimizer(model, config)
    final_rows = []
    refit_epochs = config.refit_max_epochs if config.vae_profile == "fmi_f1_sig" else best_epoch
    refit_best_value, refit_best_state, refit_stale = np.inf, None, 0
    refit_progress = trange(
        1, refit_epochs + 1, desc="TreeVAE refit full T", unit="epoch", leave=True,
        dynamic_ncols=True,
    )
    for epoch in refit_progress:
        metrics = _run_tree_epoch(
            model, full_loader, optimizer, True, epoch, config, device, target_distance,
            X_validation, y_validation,
        )
        final_rows.append({
            "epoch": epoch,
            "beta": beta_schedule(epoch, config),
            **{f"full_T_{key}": value for key, value in metrics.items()},
        })
        if config.vae_profile == "fmi_f1_sig":
            refit_monitor = (
                metrics["root"] + 0.5 * (metrics["left_child"] + metrics["right_child"])
                + metrics.get("signature_count", 0.0)
                + metrics.get("count_leaves", 0.0)
                + 0.5 * metrics.get("f1_loss", 0.0)
                + 0.5 * metrics.get("metric", 0.0)
            )
            final_rows[-1]["full_T_monitor"] = refit_monitor
            if refit_monitor < refit_best_value - config.refit_min_delta:
                refit_best_value = refit_monitor
                refit_best_state = copy.deepcopy(model.state_dict())
                refit_stale = 0
            else:
                refit_stale += 1
        refit_progress.set_postfix(loss=f"{metrics['total']:.4f}", metric=f"{metrics['metric']:.4f}")
        if (
            config.vae_profile == "fmi_f1_sig"
            and epoch >= config.refit_min_epochs
            and refit_stale >= config.refit_patience
        ):
            print(
                f"VAE_fmi_f1_sig refit early stopping: epoch={epoch}, "
                f"best_monitor={refit_best_value:.6f}"
            )
            break
    refit_progress.close()
    if refit_best_state is not None:
        model.load_state_dict(refit_best_state)
    model.eval()
    final_history = pd.DataFrame(final_rows)
    final_history.to_csv(output_dir / "vae_final_train_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "vae_final_train_state.pt")
    print(f"TreeVAE finale addestrato | best epoch interno={best_epoch} | {time.time() - started:.1f}s")
    return model, best_epoch, early_history, final_history, structural_np


def train_vae(
    corpus: CorpusBundle,
    config: VAEConfig,
    seed: int,
    output_dir: Path,
    data: ProcessedDataset | None = None,
    rulecard_config: RuleCardConfig | None = None,
    source_dir: Path | str | None = None,
) -> tuple[torch.nn.Module, int, pd.DataFrame, pd.DataFrame, np.ndarray]:
    if getattr(config, "representation", "row") == "tree":
        if data is None:
            raise ValueError("vae.representation='tree' richiede data.")
        return _train_tree_vae(corpus, config, seed, output_dir, data, rulecard_config, source_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"VAE | device={device.type} | latent_dim={config.latent_dim}")
    all_idx = np.arange(corpus.n_sc)
    train_idx, holdout_idx = train_test_split(all_idx, test_size=0.20, random_state=seed)
    train_loader = DataLoader(ScoreCardDataset(corpus, train_idx), batch_size=config.batch_size, shuffle=True)
    holdout_loader = DataLoader(ScoreCardDataset(corpus, holdout_idx), batch_size=config.batch_size, shuffle=False)
    full_loader = DataLoader(ScoreCardDataset(corpus, all_idx), batch_size=config.batch_size, shuffle=True)
    print(f"Calcolo D_structural | mode={config.structural_distance_mode}")
    structural_np = _training_structural_distance(corpus, config, data)
    target_np, predictive_np = _metric_target_distance(
        corpus, config, structural_np, data, rulecard_config, source_dir
    )
    target_distance = torch.tensor(target_np, dtype=torch.float32, device=device)
    print("VAE metric target")
    print(f"  metric_target={config.metric_target} | metric_alpha={config.metric_alpha:.3f}")
    print(f"  structural_distance_mode={config.structural_distance_mode}")
    print(f"  early_stopping_monitor={config.early_stopping_monitor}")
    if config.early_stopping_monitor == "decoded_composite":
        print(
            "  decoded_monitor="
            f"metric + {config.decoded_reconstruction_weight:.3f}*reconstruction_ratio + "
            f"{config.decoded_predictive_weight:.3f}*(1-classification_fidelity) | "
            f"every={config.decoded_monitor_interval} | min_epochs={config.decoded_monitor_min_epochs} | "
            f"cards={config.decoded_monitor_cards}"
        )
    print(f"  metric_loss_latent=mu | diagonal_excluded=True | normalization=zscore_offdiag")
    print("  " + _distance_stats("D_structural", structural_np))
    if predictive_np is not None:
        print("  " + _distance_stats("D_predictive", predictive_np))
    print("  " + _distance_stats("D_target", target_np))
    counts = np.bincount(corpus.tree_rows_df.query("is_padding == 0")["pattern_id"].to_numpy(), minlength=corpus.n_pat).astype(np.float32)
    pattern_weight = torch.tensor(
        (counts.sum() / np.maximum(counts, 1.0)) ** 0.5,
        dtype=torch.float32,
        device=device,
    )

    torch.manual_seed(seed)
    model = build_vae(corpus, config, device)
    optimizer = _make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=config.lr_patience, min_lr=config.min_lr)
    history = []
    best_val, best_epoch, best_state, no_improvement = np.inf, 0, None, 0
    reconstruction_reference: float | None = None
    decoded_positions = np.asarray(
        holdout_idx[: min(config.decoded_monitor_cards, len(holdout_idx))], dtype=np.int64
    )
    started = time.time()
    progress = trange(
        1, config.epochs + 1, desc="VAE early stopping", unit="epoch", leave=True,
        dynamic_ncols=True,
    )
    for epoch in progress:
        train_metrics = _run_epoch(model, train_loader, optimizer, True, epoch, config, device, pattern_weight, target_distance)
        val_metrics = _run_epoch(model, holdout_loader, None, False, epoch, config, device, pattern_weight, target_distance)
        if reconstruction_reference is None:
            reconstruction_reference = max(val_metrics["reconstruction"], 1e-8)
        decoded_metrics = {
            "rmse": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan,
            "positive_recall": np.nan, "fidelity": np.nan,
        }
        reconstruction_ratio = val_metrics["reconstruction"] / reconstruction_reference
        monitor_is_due = config.early_stopping_monitor != "decoded_composite" or (
            epoch >= config.decoded_monitor_min_epochs
            and epoch % config.decoded_monitor_interval == 0
        )
        if config.early_stopping_monitor == "decoded_composite":
            if monitor_is_due:
                if data is None or rulecard_config is None:
                    raise ValueError("decoded_composite richiede data e rulecard_config.")
                decoded_metrics = _decoded_predictive_rmse(
                    model, corpus, decoded_positions, data, rulecard_config, source_dir, config
                )
                monitored_value = (
                    val_metrics["metric"]
                    + config.decoded_reconstruction_weight * reconstruction_ratio
                    + config.decoded_predictive_weight * (1.0 - decoded_metrics["fidelity"])
                )
                scheduler.step(monitored_value)
            else:
                monitored_value = np.nan
        else:
            monitored_value = val_metrics[config.early_stopping_monitor]
            scheduler.step(monitored_value)
        row = {
            "epoch": epoch,
            "beta": beta_schedule(epoch, config),
            "lr": optimizer.param_groups[0]["lr"],
        }
        row["vae_earlystop_monitored"] = monitored_value
        row["vae_decoded_predictive_rmse"] = decoded_metrics["rmse"]
        row["vae_decoded_balanced_accuracy"] = decoded_metrics["balanced_accuracy"]
        row["vae_decoded_macro_f1"] = decoded_metrics["macro_f1"]
        row["vae_decoded_positive_recall"] = decoded_metrics["positive_recall"]
        row["vae_decoded_fidelity"] = decoded_metrics["fidelity"]
        row["vae_reconstruction_ratio"] = reconstruction_ratio
        row.update({f"vae_train_{key}": value for key, value in train_metrics.items()})
        row.update({f"vae_earlystop_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        progress.set_postfix(train=f"{train_metrics['total']:.4f}", val=f"{val_metrics['total']:.4f}", lr=f"{row['lr']:.1e}")
        if not monitor_is_due:
            continue
        if monitored_value < best_val - config.min_delta:
            best_val, best_epoch, best_state, no_improvement = monitored_value, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            no_improvement += (
                config.decoded_monitor_interval
                if config.early_stopping_monitor == "decoded_composite" else 1
            )
        if no_improvement >= config.patience:
            print(
                f"VAE early stopping: epoch={epoch}, best_epoch={best_epoch}, "
                f"monitor={config.early_stopping_monitor}, best={best_val:.6f}"
            )
            break
    progress.close()
    if best_state is None:
        raise RuntimeError("Il VAE non ha prodotto uno stato valido.")
    early_history = pd.DataFrame(history)
    early_history.to_csv(output_dir / "vae_earlystop_history.csv", index=False)
    torch.save(best_state, output_dir / "vae_earlystop_state.pt")

    # Refit finale su tutte le scorecard T per l'epoca selezionata internamente.
    torch.manual_seed(seed)
    model = build_vae(corpus, config, device)
    optimizer = _make_optimizer(model, config)
    final_rows = []
    refit_progress = trange(
        1, best_epoch + 1, desc="VAE refit full T", unit="epoch", leave=True,
        dynamic_ncols=True,
    )
    for epoch in refit_progress:
        metrics = _run_epoch(model, full_loader, optimizer, True, epoch, config, device, pattern_weight, target_distance)
        final_rows.append({
            "epoch": epoch,
            "beta": beta_schedule(epoch, config),
            **{f"full_T_{k}": v for k, v in metrics.items()},
        })
        refit_progress.set_postfix(loss=f"{metrics['total']:.4f}", metric=f"{metrics['metric']:.4f}")
    refit_progress.close()
    model.eval()
    final_history = pd.DataFrame(final_rows)
    final_history.to_csv(output_dir / "vae_final_train_history.csv", index=False)
    torch.save(model.state_dict(), output_dir / "vae_final_train_state.pt")
    print(f"VAE finale addestrato | best epoch interno={best_epoch} | {time.time() - started:.1f}s")
    return model, best_epoch, early_history, final_history, structural_np


@torch.no_grad()
def encode_all(
    model: torch.nn.Module,
    corpus: CorpusBundle,
    batch_size: int,
    data: ProcessedDataset | None = None,
) -> np.ndarray:
    device = next(model.parameters()).device
    idx = np.arange(corpus.n_sc)
    if getattr(model, "n_trees", None) is not None and hasattr(model, "n_features"):
        if data is None:
            raise ValueError("encode_all su TreeVAE richiede data.")
        targets = build_tree_matrix_targets(corpus, data)
        loader = DataLoader(TreeScoreCardDataset(targets, idx), batch_size=batch_size, shuffle=False)
        model.eval()
        embeddings = []
        for batch in loader:
            matrix, prior = batch["x"].to(device), batch["prior"].to(device)
            mu, _ = model.encode(matrix, prior)
            embeddings.append(mu.cpu().numpy())
        return np.vstack(embeddings)
    loader = DataLoader(ScoreCardDataset(corpus, idx), batch_size=batch_size, shuffle=False)
    model.eval()
    embeddings = []
    for batch in loader:
        matrix, prior = batch[1].to(device), batch[2].to(device)
        mu, _ = model.encode(matrix, prior)
        embeddings.append(mu.cpu().numpy())
    return np.vstack(embeddings)
