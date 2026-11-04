from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import time
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..core.artifacts import ProcessedDataset
from ..core.config import GeneticConfig, RuleCardConfig
from .scorecards import (
    extract_leaves,
    import_rulecard,
    scorecard_from_leaves,
    scorecard_outputs,
    valid_rulecard,
)


def _variant_catalog(config: RuleCardConfig, seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    variant_id = 0
    for sample_mode in config.sample_modes:
        for learning_rate in config.learning_rates:
            for top_k_features in config.top_k_features:
                for reuse_features in config.reuse_features:
                    for feature_fraction in config.feature_fractions:
                        rows.append({
                            "variant_id": variant_id,
                            "sample_mode": sample_mode,
                            "learning_rate": float(learning_rate),
                            "top_k_features": top_k_features,
                            "reuse_features": bool(reuse_features),
                            "feature_fraction": float(feature_fraction),
                        })
                        variant_id += 1
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise ValueError("La griglia varianti RuleCard e' vuota.")
    rng = np.random.default_rng(seed)
    replace_variants = config.n_cards > len(catalog)
    if replace_variants:
        # Ripete cicli completi e indipendentemente rimescolati della griglia.
        # Cosi' anche i corpus grandi coprono le varianti in modo bilanciato;
        # seed, campione di train e subset di feature cambiano a ogni tentativo.
        full_cycles, remainder = divmod(config.n_cards, len(catalog))
        cycles = [rng.permutation(len(catalog)) for _ in range(full_cycles)]
        if remainder:
            cycles.append(rng.permutation(len(catalog))[:remainder])
        selected = np.concatenate(cycles)
    else:
        selected = rng.choice(len(catalog), size=config.n_cards, replace=False)
    selected_catalog = catalog.iloc[selected].reset_index(drop=True)
    selected_catalog["variant_slot"] = np.arange(len(selected_catalog))
    selected_catalog["variant_replace"] = bool(replace_variants)
    return selected_catalog


def _scorecard_signature(leaves: pd.DataFrame, decimals: int = 10) -> str:
    rows = []
    ordered = leaves.sort_values(["tree_idx", "rule", "w"]).reset_index(drop=True)
    for row in ordered.itertuples(index=False):
        path = tuple(
            (int(feature_idx), round(float(threshold), decimals), str(operator))
            for feature_idx, threshold, operator in row.conditions_key
        )
        rows.append((int(row.tree_idx), path, round(float(row.w), decimals)))
    return repr(tuple(rows))


def _sample_training_data(
    data: ProcessedDataset,
    sample_mode: str,
    rng: np.random.Generator,
    bootstrap_fraction: float,
    stratified: bool = True,
    balance_classes: bool = False,
) -> np.ndarray:
    n_train = len(data.X_train)
    if sample_mode == "full_train":
        return np.arange(n_train)
    if sample_mode == "bootstrap":
        fraction, replace = bootstrap_fraction, True
    elif sample_mode.startswith("majority_"):
        fraction = float(sample_mode.split("_", 1)[1]) / 100.0
        replace = False
    elif sample_mode.startswith("subsample_"):
        # Compatibilita con gli artifact storici: questi nomi indicavano gia'
        # un campionamento bilanciato calcolato sulla classe maggioritaria.
        fraction = float(sample_mode.split("_", 1)[1]) / 100.0
        replace = False
    else:
        raise ValueError(f"sample_mode non supportato: {sample_mode}")

    if stratified:
        labels = np.unique(data.y_train)
        class_indices_by_label = [np.flatnonzero(data.y_train == label) for label in labels]
        if balance_classes:
            if sample_mode == "bootstrap":
                total_size = max(len(labels), int(round(fraction * n_train)))
                base_size, remainder = divmod(total_size, len(labels))
                sizes = [base_size + int(index < remainder) for index in range(len(labels))]
            else:
                target_per_class = max(
                    1,
                    int(round(fraction * max(len(indices) for indices in class_indices_by_label))),
                )
                sizes = [target_per_class] * len(class_indices_by_label)
            sampled_parts = [
                rng.choice(indices, size=size, replace=replace or size > len(indices))
                for indices, size in zip(class_indices_by_label, sizes)
            ]
            sampled = np.concatenate(sampled_parts).astype(np.int64, copy=False)
            rng.shuffle(sampled)
            return sampled

        sampled_parts = []
        for class_indices in class_indices_by_label:
            size = max(1, int(round(fraction * len(class_indices))))
            sampled_parts.append(rng.choice(class_indices, size=size, replace=replace))
        sampled = np.concatenate(sampled_parts).astype(np.int64, copy=False)
        rng.shuffle(sampled)
        return sampled

    if sample_mode == "bootstrap":
        size = int(round(bootstrap_fraction * n_train))
        return rng.choice(n_train, size=size, replace=True)
    size = max(2, int(round(fraction * n_train)))
    return rng.choice(n_train, size=size, replace=False)


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label=1) -> float:
    true_positive = y_true == positive_label
    predicted_positive = y_pred == positive_label
    tp = float(np.logical_and(true_positive, predicted_positive).sum())
    fp = float(np.logical_and(~true_positive, predicted_positive).sum())
    fn = float(np.logical_and(true_positive, ~predicted_positive).sum())
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0.0 else 2.0 * tp / denominator


def _initial_selection_score(accuracy: float, f1: float, config: RuleCardConfig) -> float:
    if config.initial_selection_metric == "accuracy":
        return accuracy
    if config.initial_selection_metric == "f1":
        return f1
    weight = config.initial_selection_accuracy_weight
    return weight * accuracy + (1.0 - weight) * f1


def _select_features(n_features: int, feature_fraction: float, rng: np.random.Generator) -> np.ndarray | None:
    if feature_fraction >= 1.0:
        return None
    size = max(2, int(round(feature_fraction * n_features)))
    return np.sort(rng.choice(n_features, size=size, replace=False))


def _resolve_top_k_features(value, n_available_features: int):
    """Converte quote come ``25pct`` in un numero di feature per la singola variante."""
    if isinstance(value, str) and value.endswith("pct"):
        fraction = float(value[:-3]) / 100.0
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"Quota top_k_features non valida: {value}")
        return min(
            n_available_features,
            max(2, int(np.ceil(fraction * n_available_features))),
        )
    return value


def _remap_leaves(leaves: pd.DataFrame, selected_features: np.ndarray | None, feature_names: list[str]) -> pd.DataFrame:
    if selected_features is None:
        return leaves
    remapped = leaves.copy()

    def remap_path(path):
        return tuple((int(selected_features[int(f)]), float(t), str(op)) for f, t, op in path)

    remapped["conditions_key"] = remapped["conditions_key"].map(remap_path)
    remapped["rule"] = remapped["conditions_key"].apply(
        lambda path: " AND ".join(f"{feature_names[int(f)]} {op} {float(t):.8g}" for f, t, op in path)
    )
    return remapped


def _fit_variant_scorecard(
    data: ProcessedDataset,
    base_config: RuleCardConfig,
    variant: pd.Series,
    scorecard_id: int,
    seed: int,
    source_dir: Path | str | None,
):
    rule_tree_regressor, rulecard_gam, scorecard_cls = import_rulecard(source_dir)
    rng = np.random.default_rng(seed)
    row_idx = _sample_training_data(
        data, str(variant["sample_mode"]), rng, base_config.bootstrap_fraction,
        base_config.stratified_sampling, base_config.balance_sampled_classes,
    )
    selected_features = _select_features(data.X_train.shape[1], float(variant["feature_fraction"]), rng)
    X_train_full = data.X_train[row_idx]
    X_train = X_train_full if selected_features is None else X_train_full[:, selected_features]
    y_train = data.y_train[row_idx]
    X_val = data.X_validation if selected_features is None else data.X_validation[:, selected_features]
    feature_names = data.feature_names if selected_features is None else [data.feature_names[int(i)] for i in selected_features]
    top_k_features = _resolve_top_k_features(
        variant["top_k_features"], X_train.shape[1]
    )
    if top_k_features == "all":
        top_k_features = np.inf
    rcgam = rulecard_gam(
        **base_config.as_rulecard_params(
            learning_rate=float(variant["learning_rate"]),
            top_k_features=top_k_features,
            reuse_features=bool(variant["reuse_features"]),
        ),
        base_estimator=rule_tree_regressor(max_depth=base_config.tree_max_depth),
        random_state=seed,
        verbose=False,
    )
    rcgam.fit(X_train, y_train)
    scorecard = scorecard_cls.from_rulecard(
        rcgam, X_train, PDO=base_config.pdo, odds0=base_config.odds0, score0=base_config.score0,
    )
    leaves = extract_leaves(rcgam, X_train, feature_names, scorecard_id, scorecard_cls)
    leaves = _remap_leaves(leaves, selected_features, data.feature_names)
    return rcgam, scorecard, scorecard_cls, leaves, base_config, X_train, X_train_full, X_val, selected_features, row_idx


def _fit_variant_process_task(task):
    data, worker_config, variant, slot, current_seed, source_dir = task
    try:
        result = _fit_variant_scorecard(
            data, worker_config, variant, slot, current_seed, source_dir
        )
        return slot, current_seed, variant, result, None
    except Exception as error:
        return slot, current_seed, variant, None, error


def build_rulecard_corpus(
    data: ProcessedDataset,
    config: RuleCardConfig,
    seed: int,
    source_dir: Path | str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Genera il corpus T usando varianti RuleCard e train sampling sul train set."""
    all_cards: list[pd.DataFrame] = []
    meta: list[dict] = []
    failures: list[dict] = []
    signatures: set[str] = set()
    accepted = 0
    attempt = 0
    started = time.time()
    max_attempts = config.n_cards * config.max_attempts_multiplier
    variants = _variant_catalog(config, seed)
    progress = tqdm(
        total=config.n_cards, desc="Corpus RuleCard", unit="scorecard", leave=True,
        dynamic_ncols=True,
    )
    # Il parallelismo utile è tra ScoreCard indipendenti. Ogni fit interno usa
    # un solo core, evitando 128 piccoli pool annidati sul dataset Breast.
    corpus_workers = max(1, min(int(config.n_jobs), config.n_cards))
    worker_config = replace(config, n_jobs=1)
    pending_slots = list(range(config.n_cards))

    with ProcessPoolExecutor(max_workers=corpus_workers) as executor:
        while pending_slots and attempt < max_attempts:
            batch_size = min(corpus_workers, len(pending_slots), max_attempts - attempt)
            batch_slots = [pending_slots.pop(0) for _ in range(batch_size)]
            tasks = []
            for slot in batch_slots:
                tasks.append((
                    data, worker_config, variants.iloc[slot], slot,
                    seed + attempt, source_dir,
                ))
                attempt += 1

            for slot, current_seed, variant, result, fit_error in executor.map(
                _fit_variant_process_task, tasks
            ):
                if fit_error is not None:
                    failures.append({
                        "candidate_id": slot, "seed": current_seed,
                        "reason": type(fit_error).__name__, "error": str(fit_error),
                        "variant_id": int(variant["variant_id"]),
                    })
                    pending_slots.append(slot)
                    continue
                try:
                    assert result is not None
                    (
                        rcgam, sc_initial, scorecard_cls, leaves, variant_config, X_fit,
                        X_fit_full, X_val, selected_features, row_idx,
                    ) = result
                    if not valid_rulecard(leaves):
                        pending_slots.append(slot)
                        continue
                    signature = _scorecard_signature(leaves)
                    if signature in signatures:
                        failures.append({
                            "candidate_id": slot, "seed": current_seed, "reason": "duplicate",
                            "variant_id": int(variant["variant_id"]),
                        })
                        pending_slots.append(slot)
                        continue
                    prior = float(rcgam.base_log_odds)
                    _, val_pred = scorecard_outputs(sc_initial, X_val)
                    validation_accuracy = float((val_pred == data.y_validation).mean())
                    validation_f1 = _binary_f1(data.y_validation, val_pred, data.model_classes[-1])
                    positive_prediction_rate = float(np.mean(val_pred == data.model_classes[-1]))
                    train_proba, train_pred = scorecard_outputs(sc_initial, X_fit)
                    rebuilt = scorecard_from_leaves(
                        leaves, prior, np.asarray(rcgam.classes_), variant_config, scorecard_cls
                    )
                    rebuild_proba, rebuild_pred = scorecard_outputs(rebuilt, X_fit_full)
                    same_pred = np.array_equal(train_pred, rebuild_pred)
                    same_proba = np.allclose(train_proba, rebuild_proba, atol=1e-12, rtol=0.0)
                    if not (same_pred and same_proba):
                        failures.append({
                            "candidate_id": slot, "seed": current_seed,
                            "reason": "roundtrip_mismatch",
                            "variant_id": int(variant["variant_id"]),
                            "same_train_prediction": bool(same_pred),
                            "same_train_probabilities": bool(same_proba),
                        })
                        pending_slots.append(slot)
                        continue
                    signatures.add(signature)
                    all_cards.append(leaves)
                    meta.append({
                "scorecard_id": slot,
                "variant_id": int(variant["variant_id"]),
                "variant_slot": int(variant["variant_slot"]),
                "seed": current_seed,
                "prior_log_odds": prior,
                "n_trees": int(leaves["tree_idx"].nunique()),
                "n_leaves": int(len(leaves)),
                "validation_accuracy": validation_accuracy,
                "validation_f1": validation_f1,
                "validation_selection_score": _initial_selection_score(
                    validation_accuracy, validation_f1, config
                ),
                "validation_positive_prediction_rate": positive_prediction_rate,
                "sample_mode": str(variant["sample_mode"]),
                "sample_size": int(len(row_idx)),
                "learning_rate": float(variant["learning_rate"]),
                "top_k_features": str(variant["top_k_features"]),
                "top_k_features_resolved": _resolve_top_k_features(
                    variant["top_k_features"], X_fit.shape[1]
                ),
                "reuse_features": bool(variant["reuse_features"]),
                "feature_fraction": float(variant["feature_fraction"]),
                "selected_features": None if selected_features is None else [int(i) for i in selected_features],
                "n_selected_features": int(data.X_train.shape[1] if selected_features is None else len(selected_features)),
                "rebuild_matches_scorecard_on_train": True,
                    })
                    accepted += 1
                    progress.update(1)
                    progress.set_postfix(
                        accepted=f"{accepted}/{config.n_cards}", attempts=attempt,
                        last_f1=f"{meta[-1]['validation_f1']:.4f}",
                        mode=str(variant["sample_mode"]),
                        pos_rate=f"{meta[-1]['validation_positive_prediction_rate']:.3f}",
                    )
                except Exception as error:
                    failures.append({
                        "candidate_id": slot, "seed": current_seed,
                        "reason": type(error).__name__, "error": str(error),
                        "variant_id": int(variant["variant_id"]),
                    })
                    pending_slots.append(slot)

    progress.close()
    if accepted != config.n_cards:
        raise RuntimeError(
            f"Corpus incompleto: {accepted}/{config.n_cards} scorecard valide dopo {attempt} tentativi."
        )
    corpus_df = pd.concat(all_cards, ignore_index=True).sort_values(
        ["scorecard_id", "tree_idx"]
    ).reset_index(drop=True)
    cards_df = pd.DataFrame(meta).sort_values("scorecard_id").reset_index(drop=True)
    elapsed = time.time() - started
    print(f"Corpus pronto: {accepted}/{config.n_cards} scorecard | tentativi={attempt} | {elapsed:.1f}s")
    return corpus_df, cards_df, failures


def select_best_initial_scorecard(
    corpus_df: pd.DataFrame,
    cards_df: pd.DataFrame,
    data: ProcessedDataset,
    config: RuleCardConfig,
    source_dir: Path | str | None,
    genetic_config: GeneticConfig,
    seed: int,
) -> tuple[int, float, pd.DataFrame, object]:
    from .scorecards import import_rulecard
    _, _, scorecard_cls = import_rulecard(source_dir)
    if config.initial_selection_metric == "accuracy":
        cards_df["validation_selection_score"] = cards_df["validation_accuracy"]
    elif config.initial_selection_metric == "f1":
        cards_df["validation_selection_score"] = cards_df["validation_f1"]
    else:
        weight = config.initial_selection_accuracy_weight
        cards_df["validation_selection_score"] = (
            weight * cards_df["validation_accuracy"]
            + (1.0 - weight) * cards_df["validation_f1"]
        )
    nonconstant = cards_df.loc[
        cards_df["validation_positive_prediction_rate"].between(
            0.0, 1.0, inclusive="neither"
        )
    ]
    if nonconstant.empty:
        eligible = cards_df
        print(
            "ATTENZIONE: tutte le ScoreCard del corpus predicono una sola classe; "
            "fallback alla classifica completa."
        )
    else:
        eligible = nonconstant
    if config.s1_choice in {"pareto", "min_rules"} or config.initial_selection_metric == "f1":
        # Il numero di foglie non coincide necessariamente con il numero di
        # regole visualizzabili dopo l'aggregazione effettuata da RuleCard.
        # Ricostruiamo quindi ogni candidata e usiamo il conteggio effettivo.
        effective_rule_counts = {}
        cards_by_id = cards_df.set_index("scorecard_id")
        for identifier in eligible["scorecard_id"].astype(int):
            candidate_leaves = corpus_df.loc[
                corpus_df["scorecard_id"] == identifier
            ].copy()
            candidate = scorecard_from_leaves(
                candidate_leaves,
                float(cards_by_id.loc[identifier, "prior_log_odds"]),
                data.model_classes,
                config,
                scorecard_cls,
            )
            effective_rule_counts[identifier] = int(len(candidate.scorecard))
        eligible = eligible.copy()
        eligible["n_scorecard_rules"] = (
            eligible["scorecard_id"].astype(int).map(effective_rule_counts)
        )
    if config.s1_choice == "pareto":
        # Frontiera non dominata: meno regole e F1 maggiore sono preferibili.
        ordered = eligible.sort_values(
            ["n_scorecard_rules", "validation_f1", "scorecard_id"],
            ascending=[True, False, True],
        )
        frontier_indices: list[int] = []
        best_f1 = -np.inf
        for index, candidate in ordered.iterrows():
            candidate_f1 = float(candidate["validation_f1"])
            if candidate_f1 > best_f1:
                frontier_indices.append(index)
                best_f1 = candidate_f1
        frontier = eligible.loc[frontier_indices].copy()

        # TOPSIS sulla frontiera: normalizzazione min--max e pesi applicati
        # alla matrice normalizzata prima delle distanze da ideale/anti-ideale.
        f1_span = max(
            float(frontier["validation_f1"].max() - frontier["validation_f1"].min()),
            1e-12,
        )
        rules_span = max(
            float(frontier["n_scorecard_rules"].max() - frontier["n_scorecard_rules"].min()),
            1e-12,
        )
        f1_normalized = (
            frontier["validation_f1"] - float(frontier["validation_f1"].min())
        ) / f1_span
        simplicity_normalized = (
            float(frontier["n_scorecard_rules"].max())
            - frontier["n_scorecard_rules"]
        ) / rules_span
        f1_weight = float(config.s1_f1_weight)
        simplicity_weight = 1.0 - f1_weight
        f1_weighted = f1_weight * f1_normalized
        simplicity_weighted = simplicity_weight * simplicity_normalized
        frontier["topsis_distance_positive"] = np.sqrt(
            (f1_weight - f1_weighted).pow(2)
            + (simplicity_weight - simplicity_weighted).pow(2)
        )
        frontier["topsis_distance_negative"] = np.sqrt(
            f1_weighted.pow(2) + simplicity_weighted.pow(2)
        )
        denominator = (
            frontier["topsis_distance_positive"]
            + frontier["topsis_distance_negative"]
        )
        frontier["topsis_score"] = np.where(
            denominator > 1e-12,
            frontier["topsis_distance_negative"] / denominator,
            1.0,
        )
        ranked = frontier.sort_values(
            ["topsis_score", "validation_f1", "n_scorecard_rules", "scorecard_id"],
            ascending=[False, False, True, True],
        )
    elif config.s1_choice == "min_rules":
        ranked = eligible.sort_values(
            ["n_scorecard_rules", "validation_selection_score", "scorecard_id"],
            ascending=[True, False, True],
        )
    elif config.initial_selection_metric == "f1":
        ranked = eligible.sort_values(
            ["validation_selection_score", "n_scorecard_rules", "scorecard_id"],
            ascending=[False, True, True],
        )
    else:
        ranked = eligible.sort_values(
            ["validation_selection_score", "validation_accuracy", "scorecard_id"],
            ascending=[False, False, True],
        )
    row = ranked.iloc[0]
    identifier = int(row["scorecard_id"])
    leaves = corpus_df.loc[corpus_df["scorecard_id"] == identifier].copy()
    scorecard = scorecard_from_leaves(
        leaves, float(row["prior_log_odds"]), data.model_classes, config, scorecard_cls
    )
    return identifier, float(row["validation_accuracy"]), leaves, scorecard
