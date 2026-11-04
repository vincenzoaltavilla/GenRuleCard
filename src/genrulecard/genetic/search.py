from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import trange

from ..core.artifacts import GeneticResult, ProcessedDataset
from ..core.config import GeneticConfig
from ..rulecard_fit.scorecards import import_rulecard, scorecard_from_leaves, scorecard_outputs
from ..vae.decoder import ConstrainedDecoder
from ..fairness import fairness_penalty, group_fairness


class MinimumRuleTargetNotReached(RuntimeError):
    """Il GA non ha prodotto alcun risultato con almeno le regole target."""


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> float:
    y_true_pos = y_true == positive_label
    y_pred_pos = y_pred == positive_label
    tp = float(np.logical_and(y_true_pos, y_pred_pos).sum())
    fp = float(np.logical_and(~y_true_pos, y_pred_pos).sum())
    fn = float(np.logical_and(y_true_pos, ~y_pred_pos).sum())
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0.0 else float(2.0 * tp / denominator)


def _mean_true_class_probability(probabilities: np.ndarray, y_true: np.ndarray, classes: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        positive = classes[-1]
        return float(np.where(y_true == positive, probabilities, 1.0 - probabilities).mean())
    class_to_column = {label: idx for idx, label in enumerate(classes)}
    columns = np.asarray([class_to_column[label] for label in y_true], dtype=np.int64)
    return float(probabilities[np.arange(len(y_true)), columns].mean())


def _rule_mask(conditions: list[tuple], X: np.ndarray) -> np.ndarray:
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


def _redundancy_penalty(scorecard, X: np.ndarray) -> float:
    rules = getattr(scorecard, "scorecard", [])
    masks = []
    for _, conditions, _, points in rules:
        if int(points) == 0:
            continue
        mask = _rule_mask(conditions, X)
        if mask.any():
            masks.append(mask)
    if len(masks) <= 1:
        return 0.0

    overlaps = []
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            union = np.logical_or(masks[left], masks[right]).sum()
            if union == 0:
                continue
            intersection = np.logical_and(masks[left], masks[right]).sum()
            overlaps.append(float(intersection / union))
    return float(np.mean(overlaps)) if overlaps else 0.0


def performance_score(
    accuracy: float,
    f1: float,
    metric: str,
    accuracy_weight: float = 0.5,
) -> float:
    if metric == "accuracy":
        return accuracy
    if metric == "f1":
        return f1
    return accuracy_weight * accuracy + (1.0 - accuracy_weight) * f1


def _rule_distance(n_rules: int, reference_n_rules: int) -> float:
    reference_n_rules = max(1, int(reference_n_rules))
    return float(abs(max(1, int(n_rules)) - reference_n_rules) / reference_n_rules)


def _rule_zone_penalty(n_rules: int, reference_n_rules: int, tolerance: float) -> float:
    return float(max(0.0, _rule_distance(n_rules, reference_n_rules) - max(0.0, float(tolerance))))


def _target_reduction_reward(
    n_rules: int,
    reference_n_rules: int,
    target_fraction: float,
) -> float:
    reference = float(max(1, int(reference_n_rules)))
    target = reference * float(target_fraction)
    denominator = max(reference - target, 1e-12)
    reward = (reference - float(max(1, int(n_rules)))) / denominator
    return float(np.clip(reward, 0.0, 1.0))


def _requested_rule_target(reference_n_rules: int, f1_weight: float) -> int:
    """Target proporzionale: complexity=1-w richiede quella riduzione."""
    reference = max(1, int(reference_n_rules))
    return max(1, int(np.floor(float(f1_weight) * reference + 0.5)))


def _f1_complexity_score(
    f1: float,
    n_rules: int,
    reference_n_rules: int,
    f1_weight: float,
    rule_tolerance: float,
    rule_objective: str,
    target_fraction: float,
) -> float:
    if rule_objective == "absolute_min_rules":
        relative_complexity = float(
            max(1, int(n_rules)) / max(1, int(reference_n_rules))
        )
        simplicity = 1.0 - float(np.clip(relative_complexity, 0.0, 1.0))
        return float(f1_weight * f1 + (1.0 - f1_weight) * simplicity)
    if rule_objective == "target_constraint":
        # complexity=0: protocollo puro F1, senza alcun vincolo sulle regole.
        if float(f1_weight) >= 1.0 - 1e-12:
            return float(f1)
        reference = max(1, int(reference_n_rules))
        target = _requested_rule_target(reference, f1_weight)
        rules = max(1, int(n_rules))
        # Target come minimo ammesso: target, target+1, ... senza tetto.
        # Prima vince il minor numero di regole ammissibile, poi l'F1.
        f1_tiebreak_scale = 0.5
        if rules >= target:
            distance_above_target = float(rules - target)
            return float(
                1_000_000.0 - distance_above_target + f1_tiebreak_scale * f1
            )
        # Candidati sotto target possono guidare la ricerca, ma restano
        # sempre sotto qualunque candidato ammissibile.
        range_violation = float(target - rules)
        return float(-range_violation + f1_tiebreak_scale * f1)
    if rule_objective == "target_reduction":
        reduction_reward = _target_reduction_reward(
            n_rules, reference_n_rules, target_fraction
        )
        return float(
            f1_weight * f1 + (1.0 - f1_weight) * reduction_reward
        )
    if rule_objective == "min_rules":
        complexity_penalty = float(max(1, int(n_rules)) / max(1, int(reference_n_rules)))
    else:
        complexity_penalty = _rule_zone_penalty(
            n_rules, reference_n_rules, rule_tolerance
        )
    return float(f1_weight * f1 - (1.0 - f1_weight) * complexity_penalty)


def _scorecard_fitness(
    scorecard,
    data: ProcessedDataset,
    config: GeneticConfig,
    validation_folds: list[np.ndarray],
    reference_n_rules: int,
    fairness_context: dict | None = None,
):
        probabilities, prediction = scorecard_outputs(scorecard, data.X_validation)
        accuracy = float((prediction == data.y_validation).mean())
        f1 = _binary_f1(data.y_validation, prediction, data.model_classes[-1])
        confidence = _mean_true_class_probability(probabilities, data.y_validation, data.model_classes)
        n_rules = int(len(scorecard.scorecard))
        reference_n_rules = max(1, int(reference_n_rules))
        rule_distance = _rule_distance(n_rules, reference_n_rules)
        if config.f1_complexity_rule_objective == "target_constraint":
            target_rules = _requested_rule_target(
                reference_n_rules, config.f1_complexity_f1_weight
            )
            if n_rules >= target_rules:
                complexity = float((n_rules - target_rules) / reference_n_rules)
            else:
                complexity = float((target_rules - n_rules) / reference_n_rules)
        elif config.f1_complexity_rule_objective == "target_reduction":
            complexity = 1.0 - _target_reduction_reward(
                n_rules, reference_n_rules,
                config.f1_complexity_target_fraction,
            )
        elif config.f1_complexity_rule_objective in {"min_rules", "absolute_min_rules"}:
            complexity = float(max(1, n_rules) / reference_n_rules)
        else:
            complexity = _rule_zone_penalty(
                n_rules, reference_n_rules, config.f1_complexity_rule_tolerance
            )
        relative_complexity = float(max(1, n_rules) / reference_n_rules)
        redundancy = _redundancy_penalty(scorecard, data.X_validation)
        if config.fitness_metric == "f1_complexity":
            performance = _f1_complexity_score(
                f1, n_rules, reference_n_rules, config.f1_complexity_f1_weight,
                config.f1_complexity_rule_tolerance,
                config.f1_complexity_rule_objective,
                config.f1_complexity_target_fraction,
            )
        else:
            performance = performance_score(
                accuracy, f1, config.fitness_metric, config.accuracy_f1_accuracy_weight
            )
        fold_performance = []
        for fold_idx in validation_folds:
            fold_accuracy = float((prediction[fold_idx] == data.y_validation[fold_idx]).mean())
            fold_f1 = _binary_f1(
                data.y_validation[fold_idx], prediction[fold_idx], data.model_classes[-1]
            )
            if config.fitness_metric == "f1_complexity":
                fold_performance.append(_f1_complexity_score(
                    fold_f1, n_rules, reference_n_rules, config.f1_complexity_f1_weight,
                    config.f1_complexity_rule_tolerance,
                    config.f1_complexity_rule_objective,
                    config.f1_complexity_target_fraction,
                ))
            else:
                fold_performance.append(performance_score(
                    fold_accuracy, fold_f1, config.fitness_metric,
                    config.accuracy_f1_accuracy_weight,
                ))
        fold_performance = np.asarray(fold_performance, dtype=np.float64)
        mean_fold_performance = float(fold_performance.mean())
        fold_std = float(fold_performance.std(ddof=0))
        stability_penalty = float(config.validation_stability_weight * fold_std)
        robust_performance = float(mean_fold_performance - stability_penalty)
        if config.fitness_metric == "f1_complexity":
            confidence_component = 0.0
            complexity_component = 0.0
            redundancy_component = 0.0
        else:
            confidence_component = float(config.confidence_weight * confidence)
            complexity_component = float(-config.complexity_weight * complexity)
            redundancy_component = float(-config.redundancy_weight * redundancy)
        selection_fitness = (
            robust_performance
            + confidence_component
            + complexity_component
            + redundancy_component
        )
        fairness_metrics = {}
        fairness_component = 0.0
        if fairness_context is not None:
            fairness_metrics = group_fairness(
                data.y_validation, prediction, fairness_context["sensitive_validation"]
            )
            if fairness_context.get("mode") == "three_objective_eod":
                # Esperimento fairness della tesi: tre obiettivi di beneficio
                # sulla stessa scala [0, 1], senza riferimenti a S1 o alla
                # distribuzione del corpus.  Il massimo numero di regole e'
                # imposto dalla rappresentazione (15 alberi * 4 foglie = 60).
                simplicity = float(np.clip(
                    1.0 - n_rules / float(fairness_context["max_rules"]),
                    0.0, 1.0,
                ))
                fairness_score = float(np.clip(
                    1.0 - fairness_metrics["equalized_odds_difference"],
                    0.0, 1.0,
                ))
                objective_weights = fairness_context["objective_weights"]
                f1_weight, simplicity_weight, fairness_weight = map(
                    float, objective_weights
                )
                selection_fitness = float(
                    f1_weight * f1
                    + simplicity_weight * simplicity
                    + fairness_weight * fairness_score
                )
                # Il termine e' registrato separatamente soltanto per audit;
                # selection_fitness contiene gia' l'intera fitness tripartita.
                fairness_component = float(fairness_weight * fairness_score)
                fairness_metrics.update({
                    "simplicity_score": simplicity,
                    "fairness_score": fairness_score,
                    "fairness_penalty": float(1.0 - fairness_score),
                    "fairness_dpd_violation": np.nan,
                    "fairness_eodd_violation": float(
                        fairness_metrics["equalized_odds_difference"]
                    ),
                })
                fairness_eligible = bool(np.unique(prediction).size >= 2)
                fairness_metrics["fairness_eligible"] = fairness_eligible
                if not fairness_eligible:
                    # Equalized odds is trivially zero for a constant
                    # classifier. Such candidates are not admissible in the
                    # fairness experiment, consistently with corpus creation.
                    selection_fitness = float("-inf")
            else:
                # Modalita' legacy, preservata per rendere riproducibili gli
                # eventuali esperimenti fairness precedenti.
                fairness_metrics.update(fairness_penalty(
                    fairness_metrics, fairness_context["reference"]
                ))
                fairness_component = float(
                    -fairness_context["penalty_weight"] * fairness_metrics["fairness_penalty"]
                )
                selection_fitness += fairness_component
        metrics = {
            "selection_fitness": float(selection_fitness),
            "validation_accuracy": accuracy,
            "validation_f1": f1,
            "validation_performance": performance,
            "fold_performance_mean": mean_fold_performance,
            "fold_performance_std": fold_std,
            "fold_performance_min": float(fold_performance.min()),
            "robust_validation_performance": robust_performance,
            "confidence_tiebreak": confidence,
            "confidence_component": confidence_component,
            "complexity_penalty": complexity,
            "rule_distance": rule_distance,
            "rule_tolerance": config.f1_complexity_rule_tolerance,
            "relative_complexity": relative_complexity,
            "complexity_component": complexity_component,
            "reference_n_scorecard_rules": reference_n_rules,
            "redundancy_penalty": redundancy,
            "redundancy_component": redundancy_component,
            "stability_penalty": stability_penalty,
            "n_scorecard_rules": n_rules,
            "fairness_objective": (
                fairness_context["feature"] if fairness_context is not None else None
            ),
            "fairness_component": fairness_component,
            "target_n_scorecard_rules": (
                None
                if config.f1_complexity_rule_objective == "target_constraint"
                and config.f1_complexity_f1_weight >= 1.0 - 1e-12
                else _requested_rule_target(
                    reference_n_rules, config.f1_complexity_f1_weight
                )
            ),
        }
        metrics.update(fairness_metrics)
        return float(selection_fitness), metrics


def _fitness(
    z: np.ndarray,
    decoder: ConstrainedDecoder,
    data: ProcessedDataset,
    config: GeneticConfig,
    validation_folds: list[np.ndarray],
    reference_n_rules: int,
    fairness_context: dict | None = None,
):
    try:
        decode_target = (
            None
            if config.f1_complexity_rule_objective == "absolute_min_rules"
            else reference_n_rules
        )
        if (
            config.f1_complexity_rule_objective == "target_constraint"
            and config.f1_complexity_f1_weight < 1.0 - 1e-12
        ):
            decode_target = _requested_rule_target(
                reference_n_rules, config.f1_complexity_f1_weight
            )
        candidate = decoder.decode(z, target_n_rules=decode_target)
        selection_fitness, metrics = _scorecard_fitness(
            candidate.scorecard, data, config, validation_folds,
            reference_n_rules,
            fairness_context,
        )
        metrics.update({
            "decoded_n_active": candidate.info.get("n_active"),
            "decoded_n_rules_prefix": candidate.info.get("n_rules_prefix"),
            "decoded_target_n_rules": candidate.info.get("target_n_rules"),
            "decoded_predicted_active": candidate.info.get("predicted_active"),
            "decoded_count_penalty": candidate.info.get("decode_count_penalty"),
        })
        return selection_fitness, candidate, metrics
    except Exception:
        return -np.inf, None, {
            "selection_fitness": -np.inf,
            "validation_accuracy": -np.inf,
            "validation_f1": -np.inf,
            "validation_performance": -np.inf,
            "fold_performance_mean": -np.inf,
            "fold_performance_std": np.nan,
            "fold_performance_min": -np.inf,
            "robust_validation_performance": -np.inf,
            "confidence_tiebreak": np.nan,
            "complexity_penalty": np.nan,
            "redundancy_penalty": np.nan,
            "n_scorecard_rules": np.nan,
        }


def _evaluate(
    population: np.ndarray,
    decoder: ConstrainedDecoder,
    data: ProcessedDataset,
    config: GeneticConfig,
    validation_folds: list[np.ndarray],
    reference_n_rules: int,
    fairness_context: dict | None = None,
):
    decode_target = (
        None
        if config.f1_complexity_rule_objective == "absolute_min_rules"
        else reference_n_rules
    )
    if (
        config.f1_complexity_rule_objective == "target_constraint"
        and config.f1_complexity_f1_weight < 1.0 - 1e-12
    ):
        decode_target = _requested_rule_target(
            reference_n_rules, config.f1_complexity_f1_weight
        )

    try:
        candidates = decoder.decode_many(population, target_n_rules=decode_target)
    except Exception:
        # Fallback conservativo: conserva il comportamento storico e rende
        # diagnosticabile un eventuale candidato problematico.
        return _evaluate_individually(
            population, decoder, data, config, validation_folds,
            reference_n_rules, fairness_context,
        )

    def evaluate_candidate(candidate):
        try:
            selection_fitness, metrics = _scorecard_fitness(
                candidate.scorecard, data, config, validation_folds,
                reference_n_rules, fairness_context,
            )
            metrics.update({
                "decoded_n_active": candidate.info.get("n_active"),
                "decoded_n_rules_prefix": candidate.info.get("n_rules_prefix"),
                "decoded_target_n_rules": candidate.info.get("target_n_rules"),
                "decoded_predicted_active": candidate.info.get("predicted_active"),
                "decoded_count_penalty": candidate.info.get("decode_count_penalty"),
            })
            return selection_fitness, candidate, metrics
        except Exception:
            return -np.inf, None, {
                "selection_fitness": -np.inf,
                "validation_accuracy": -np.inf,
                "validation_f1": -np.inf,
                "validation_performance": -np.inf,
                "fold_performance_mean": -np.inf,
                "fold_performance_std": np.nan,
                "fold_performance_min": -np.inf,
                "robust_validation_performance": -np.inf,
                "confidence_tiebreak": np.nan,
                "complexity_penalty": np.nan,
                "redundancy_penalty": np.nan,
                "n_scorecard_rules": np.nan,
            }

    n_jobs = max(1, min(int(config.n_jobs), len(population)))
    if n_jobs == 1:
        evaluated = [evaluate_candidate(candidate) for candidate in candidates]
    else:
        # map mantiene l'ordine della popolazione, quindi non altera selezione
        # o riproducibilita del GA.
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            evaluated = list(executor.map(evaluate_candidate, candidates))
    fitness = np.asarray([item[0] for item in evaluated], dtype=np.float64)
    return fitness, evaluated


def _evaluate_individually(
    population: np.ndarray,
    decoder: ConstrainedDecoder,
    data: ProcessedDataset,
    config: GeneticConfig,
    validation_folds: list[np.ndarray],
    reference_n_rules: int,
    fairness_context: dict | None = None,
):
    evaluated = [
        _fitness(
            z, decoder, data, config, validation_folds, reference_n_rules,
            fairness_context,
        )
        for z in population
    ]
    fitness = np.asarray([item[0] for item in evaluated], dtype=np.float64)
    return fitness, evaluated


def _mean_metric(evaluated: list[tuple], key: str) -> float:
    values = [item[2][key] for item in evaluated if np.isfinite(item[2].get(key, np.nan))]
    return float(np.mean(values)) if values else np.nan


def _deterministic_order(
    fitness: np.ndarray,
    evaluated: list[tuple],
    config: GeneticConfig,
) -> np.ndarray:
    """Ordine totale e stabile, anche in presenza di fitness quasi uguali."""
    if not config.deterministic:
        return np.argsort(np.nan_to_num(fitness, nan=-np.inf))[::-1]
    decimals = int(config.fitness_round_decimals)
    ranked = []
    for index, value in enumerate(fitness):
        metrics = evaluated[index][2]
        finite = float(value) if np.isfinite(value) else -np.inf
        rounded_fitness = round(finite, decimals) if np.isfinite(finite) else -np.inf
        f1 = float(metrics.get("validation_f1", -np.inf))
        rounded_f1 = round(f1, decimals) if np.isfinite(f1) else -np.inf
        fold_std = float(metrics.get("fold_performance_std", np.inf))
        rules = float(metrics.get("n_scorecard_rules", np.inf))
        ranked.append((index, rounded_fitness, rounded_f1, fold_std, rules))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[3], item[4], item[0]))
    return np.asarray([item[0] for item in ranked], dtype=np.int64)


def _save_reproducibility_checkpoint(
    output_dir: Path,
    step: int,
    population: np.ndarray,
    fitness: np.ndarray,
    best_z: np.ndarray,
    best_fitness: float,
    no_improvement: int,
    rng: np.random.Generator,
) -> None:
    np.savez_compressed(
        output_dir / "ga_checkpoint_latest.npz",
        step=np.asarray(step, dtype=np.int64),
        population=np.asarray(population, dtype=np.float32),
        fitness=np.asarray(fitness, dtype=np.float64),
        best_z=np.asarray(best_z, dtype=np.float32),
        best_fitness=np.asarray(best_fitness, dtype=np.float64),
        no_improvement=np.asarray(no_improvement, dtype=np.int64),
        rng_state=np.asarray([rng.bit_generator.state], dtype=object),
    )


def _initial_population(
    embeddings: np.ndarray,
    decoder: ConstrainedDecoder,
    config: GeneticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Build the GA population from random corpus points or anchor neighbours."""
    n_embeddings = len(embeddings)
    if config.initial_anchor_fraction == 0.0:
        indices = rng.choice(
            n_embeddings,
            size=config.population_size,
            replace=config.population_size > n_embeddings,
        )
        return embeddings[indices].copy().astype(np.float32), -1, 0, indices.astype(np.int64)

    anchor_id = int(decoder.corpus.best_initial_id)
    try:
        anchor_idx = decoder.corpus.card_ids.index(anchor_id)
    except ValueError as error:
        raise RuntimeError(f"Best initial scorecard id={anchor_id} non presente nel corpus.") from error

    requested = max(1, int(round(config.population_size * config.initial_anchor_fraction)))
    anchor_count = min(requested, config.population_size, n_embeddings)
    anchor = embeddings[anchor_idx]
    distances = np.linalg.norm(embeddings - anchor[None, :], axis=1)
    local_indices = np.argsort(distances, kind="stable")[:anchor_count]

    random_count = config.population_size - anchor_count
    remaining_indices = np.setdiff1d(
        np.arange(n_embeddings, dtype=np.int64), local_indices, assume_unique=True
    )
    random_pool = remaining_indices if len(remaining_indices) else np.arange(n_embeddings)
    random_indices = rng.choice(
        random_pool,
        size=random_count,
        replace=random_count > len(random_pool),
    )
    indices = np.concatenate([local_indices, np.asarray(random_indices, dtype=np.int64)])
    population = embeddings[indices].copy().astype(np.float32)
    return population, anchor_idx, anchor_count, indices.astype(np.int64)


def _history_row(
    step: int,
    best_fitness: float,
    best_metrics: dict,
    evaluated: list[tuple],
    best_initial_validation_accuracy: float,
    no_improvement: int = 0,
    step_metrics: dict | None = None,
    step_best_fitness: float | None = None,
    stopped_early: bool = False,
    step0_source_index: int | None = None,
    step0_source_scorecard_id: int | None = None,
    step0_matches_best_initial_index: bool | None = None,
    step0_matches_best_initial_id: bool | None = None,
    step0_reconstruction_metrics: dict | None = None,
) -> dict:
    row = {
        "step": step,
        "best_selection_fitness": best_fitness,
        "mean_selection_fitness": _mean_metric(evaluated, "selection_fitness"),
        "best_validation_accuracy": best_metrics["validation_accuracy"],
        "mean_validation_accuracy": _mean_metric(evaluated, "validation_accuracy"),
        "best_validation_f1": best_metrics["validation_f1"],
        "mean_validation_f1": _mean_metric(evaluated, "validation_f1"),
        "best_validation_performance": best_metrics["validation_performance"],
        "best_confidence_tiebreak": best_metrics["confidence_tiebreak"],
        "best_confidence_component": best_metrics["confidence_component"],
        "best_complexity_penalty": best_metrics["complexity_penalty"],
        "best_complexity_component": best_metrics["complexity_component"],
        "best_redundancy_penalty": best_metrics["redundancy_penalty"],
        "best_redundancy_component": best_metrics["redundancy_component"],
        "best_stability_penalty": best_metrics["stability_penalty"],
        "best_n_scorecard_rules": best_metrics["n_scorecard_rules"],
        "best_fold_performance_mean": best_metrics["fold_performance_mean"],
        "best_fold_performance_std": best_metrics["fold_performance_std"],
        "best_fold_performance_min": best_metrics["fold_performance_min"],
        "best_robust_validation_performance": best_metrics["robust_validation_performance"],
        "best_initial_validation_accuracy": best_initial_validation_accuracy,
        "no_improvement_steps": no_improvement,
        "stopped_early": stopped_early,
        "step0_source_index": step0_source_index,
        "step0_source_scorecard_id": step0_source_scorecard_id,
        "step0_matches_best_initial_index": step0_matches_best_initial_index,
        "step0_matches_best_initial_id": step0_matches_best_initial_id,
        "step0_best_ga_is_decoded_best_initial": step0_matches_best_initial_index,
    }
    if step0_reconstruction_metrics:
        row.update(step0_reconstruction_metrics)
    if step_metrics is not None:
        row.update({
            "step_best_selection_fitness": step_best_fitness,
            "step_best_validation_accuracy": step_metrics["validation_accuracy"],
            "step_best_validation_f1": step_metrics["validation_f1"],
            "step_best_validation_performance": step_metrics["validation_performance"],
            "step_best_robust_validation_performance": step_metrics["robust_validation_performance"],
            "step_best_confidence_component": step_metrics["confidence_component"],
            "step_best_complexity_component": step_metrics["complexity_component"],
            "step_best_redundancy_component": step_metrics["redundancy_component"],
            "step_best_stability_penalty": step_metrics["stability_penalty"],
        })
    return row


def run_genetic_search(
    embeddings: np.ndarray,
    decoder: ConstrainedDecoder,
    data: ProcessedDataset,
    config: GeneticConfig,
    seed: int,
    best_initial_validation_accuracy: float,
    output_dir: Path,
    best_initial_scorecard=None,
    fairness_context: dict | None = None,
) -> GeneticResult:
    if config.n_jobs > 1:
        # Ogni worker decodifica un individuo; un solo thread PyTorch per
        # worker evita n_jobs x thread interni (oversubscription).
        import torch
        torch.set_num_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    absolute_mode = config.f1_complexity_rule_objective == "absolute_min_rules"
    if absolute_mode:
        # S1 non partecipa neppure come riferimento diagnostico della ricerca.
        best_initial_scorecard = None
    rng = np.random.default_rng(seed)
    requested_folds = int(config.validation_folds)
    if config.validation_folds_dynamic:
        n_validation = int(len(data.y_validation))
        _, dynamic_class_counts = np.unique(data.y_validation, return_counts=True)
        minority_count = int(dynamic_class_counts.min())
        if n_validation < 500:
            desired_folds = 3
        elif n_validation < 2500:
            desired_folds = 5
        else:
            desired_folds = 10
        # Evita fold troppo piccoli: almeno circa 10 elementi della classe
        # minoritaria e 25 osservazioni complessive per fold.
        max_by_minority = max(2, minority_count // 10)
        max_by_size = max(2, n_validation // 25)
        requested_folds = max(
            2,
            min(desired_folds, requested_folds, max_by_minority, max_by_size),
        )
        print(
            f"GA validation folds dinamici | n_validation={n_validation} | "
            f"minority={minority_count} | folds={requested_folds}"
        )
    if requested_folds == 1:
        validation_folds = [np.arange(len(data.y_validation), dtype=np.int64)]
    else:
        _, class_counts = np.unique(data.y_validation, return_counts=True)
        effective_folds = min(requested_folds, int(class_counts.min()))
        if effective_folds < 2:
            raise ValueError("La validation non contiene abbastanza esempi per fold stratificati.")
        splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        validation_folds = [
            heldout.astype(np.int64)
            for _, heldout in splitter.split(np.zeros(len(data.y_validation)), data.y_validation)
        ]
    _, latent_dim = embeddings.shape
    population, anchor_idx, anchor_count, population_source_indices = _initial_population(
        embeddings, decoder, config, rng
    )
    if config.deterministic or config.save_reproducibility_artifacts:
        np.save(output_dir / "initial_population.npy", population)
        np.save(output_dir / "initial_population_source_indices.npy", population_source_indices)
        np.save(output_dir / "validation_folds.npy", np.asarray(validation_folds, dtype=object), allow_pickle=True)
        embedding_hash = hashlib.sha256(np.ascontiguousarray(embeddings).tobytes()).hexdigest()
        (output_dir / "ga_reproducibility_manifest.json").write_text(
            json.dumps({
                "seed": int(seed),
                "deterministic": bool(config.deterministic),
                "artifacts_only": bool(
                    config.save_reproducibility_artifacts and not config.deterministic
                ),
                "fitness_round_decimals": int(config.fitness_round_decimals),
                "tie_break": "fitness_desc,f1_desc,fold_std_asc,n_rules_asc,index_asc",
                "embedding_sha256": embedding_hash,
                "population_dtype": str(population.dtype),
                "population_shape": list(population.shape),
            }, indent=2),
            encoding="utf-8",
        )

    max_representable_rules = max(1, int(decoder.corpus.t_max) * 4)
    reference_n_rules = (
        max_representable_rules
        if config.f1_complexity_rule_objective == "absolute_min_rules"
        else int(len(best_initial_scorecard.scorecard))
        if best_initial_scorecard is not None
        else max_representable_rules
    )
    initial_fitness = np.nan
    initial_metrics = None
    if best_initial_scorecard is not None:
        initial_fitness, initial_metrics = _scorecard_fitness(
            best_initial_scorecard, data, config, validation_folds,
            reference_n_rules,
            fairness_context,
        )
        if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
            print(
                f"{decoder.corpus.n_sc} scorecard iniziali: "
                f"selected_by=validation_f1 | "
                f"val_f1={initial_metrics['validation_f1']:.4f} | "
                f"n_rules={int(initial_metrics['n_scorecard_rules'])} | "
                f"ga_fit={initial_fitness:.4f}"
            )

    constraint_active = config.f1_complexity_f1_weight < 1.0 - 1e-12
    target_rules = _requested_rule_target(
        reference_n_rules, config.f1_complexity_f1_weight
    )
    if config.f1_complexity_rule_objective == "target_constraint":
        objective_detail = (
            f"requested_rule_reduction={1.0 - config.f1_complexity_f1_weight:.1%} | "
            f"target_rules={target_rules if constraint_active else 'unconstrained'} | "
            f"target_mode={'target_then_upward' if constraint_active else 'pure_f1'} | "
        )
    else:
        objective_detail = (
            f"target_fraction={config.f1_complexity_target_fraction:.3f} | "
        )
    if absolute_mode:
        print(
            "GA fitness | "
            f"{config.f1_complexity_f1_weight:.1f}*F1_val + "
            f"{1.0 - config.f1_complexity_f1_weight:.1f}*(1-rules/{reference_n_rules})"
        )
    else:
        print(
            "GA fitness | "
            f"metric={config.fitness_metric} | "
            f"f1_w={config.f1_complexity_f1_weight:.3f} | "
            f"step0_f1_w={config.f1_complexity_initial_f1_weight if config.f1_complexity_initial_f1_weight is not None else config.f1_complexity_f1_weight:.3f} | "
            f"rule_tol={config.f1_complexity_rule_tolerance:.3f} | "
            f"rule_objective={config.f1_complexity_rule_objective} | "
            f"{objective_detail}reference_rules={reference_n_rules} | "
            f"validation_folds={len(validation_folds)}"
        )
    print(
        "GA inizializzazione | "
        + (
            f"random_corpus={config.population_size}/{config.population_size}"
            if anchor_count == 0 else
            f"anchor=attivo | local_neighbors={anchor_count}/{config.population_size}"
        )
    )
    step0_config = config
    if config.fitness_metric == "f1_complexity" and config.f1_complexity_initial_f1_weight is not None:
        step0_config = replace(
            config,
            f1_complexity_f1_weight=config.f1_complexity_initial_f1_weight,
        )
    fitness, evaluated = _evaluate(
        population, decoder, data, step0_config, validation_folds, reference_n_rules,
        fairness_context,
    )
    if not np.isfinite(fitness).any():
        raise RuntimeError("Nessuna scorecard valida nella popolazione iniziale del GA.")

    # Optional audit trail used by the iterative-refinement experiment.  It is
    # deliberately enabled only for the monitored alpha (currently 0.7), so
    # retaining every generation does not multiply storage by the full
    # 11-alpha sweep.  These are the complete latent populations and their
    # already-computed fitness values; no decoding or test-set evaluation is
    # repeated for the audit.
    save_population_history = (
        (
            os.environ.get("GENRULECARD_SAVE_POPULATION_HISTORY") == "1"
            or os.environ.get("GENRULECARD_SAVE_FINAL_POPULATION") == "1"
        )
        and np.isclose(config.f1_complexity_f1_weight, 0.7, atol=1e-12)
    )
    population_history = [population.copy()] if save_population_history else None
    fitness_history = [fitness.copy()] if save_population_history else None

    best_idx = int(_deterministic_order(fitness, evaluated, config)[0])
    step0_source_index = int(population_source_indices[best_idx])
    step0_source_id = int(decoder.corpus.card_ids[step0_source_index])
    anchor_id = int(decoder.corpus.best_initial_id)
    step0_matches_anchor_index = bool(anchor_idx >= 0 and step0_source_index == anchor_idx)
    step0_matches_anchor_id = bool(step0_source_id == anchor_id)
    decoded_s1_metrics = None
    decoded_s1_fitness = np.nan
    decoded_s1_population_pos = None
    if anchor_idx >= 0:
        anchor_positions = np.flatnonzero(population_source_indices == anchor_idx)
        if len(anchor_positions):
            decoded_s1_population_pos = int(anchor_positions[0])
            decoded_s1_fitness = float(fitness[decoded_s1_population_pos])
            decoded_s1_metrics = dict(evaluated[decoded_s1_population_pos][2])
    _, _, scorecard_cls = import_rulecard(decoder.source_dir)
    source_meta_row = decoder.corpus.cards_df.loc[
        decoder.corpus.cards_df["scorecard_id"] == step0_source_id
    ].iloc[0]
    source_original_leaves = decoder.corpus.corpus_df.loc[
        decoder.corpus.corpus_df["scorecard_id"] == step0_source_id
    ].copy()
    source_original_scorecard = scorecard_from_leaves(
        source_original_leaves,
        float(source_meta_row["prior_log_odds"]),
        data.model_classes,
        decoder.config,
        scorecard_cls,
    )
    source_original_fitness, source_original_metrics = _scorecard_fitness(
        source_original_scorecard, data, step0_config, validation_folds, reference_n_rules,
        fairness_context,
    )
    step0_reconstruction_metrics = {}
    if initial_metrics is not None:
        step0_reconstruction_metrics.update({
            "step0_s1_original_fitness": initial_fitness,
            "step0_s1_original_accuracy": initial_metrics["validation_accuracy"],
            "step0_s1_original_f1": initial_metrics["validation_f1"],
            "step0_s1_original_rules": initial_metrics["n_scorecard_rules"],
        })
    if decoded_s1_metrics is not None:
        step0_reconstruction_metrics.update({
            "step0_decoded_s1_population_pos": decoded_s1_population_pos,
            "step0_decoded_s1_fitness": decoded_s1_fitness,
            "step0_decoded_s1_accuracy": decoded_s1_metrics["validation_accuracy"],
            "step0_decoded_s1_f1": decoded_s1_metrics["validation_f1"],
            "step0_decoded_s1_rules": decoded_s1_metrics["n_scorecard_rules"],
        })
    step0_reconstruction_metrics.update({
        "step0_s2_source_original_fitness": source_original_fitness,
        "step0_s2_source_original_accuracy": source_original_metrics["validation_accuracy"],
        "step0_s2_source_original_f1": source_original_metrics["validation_f1"],
        "step0_s2_source_original_rules": source_original_metrics["n_scorecard_rules"],
        "step0_s2_source_variant_id": int(source_meta_row["variant_id"]),
        "step0_s2_source_variant_slot": int(source_meta_row["variant_slot"]),
        "step0_s2_source_sample_mode": str(source_meta_row["sample_mode"]),
        "step0_s2_source_learning_rate": float(source_meta_row["learning_rate"]),
        "step0_s2_source_top_k_features": str(source_meta_row["top_k_features"]),
        "step0_s2_source_reuse_features": bool(source_meta_row["reuse_features"]),
        "step0_s2_source_feature_fraction": float(source_meta_row["feature_fraction"]),
    })
    best_z = population[best_idx].copy()
    best_fitness = float(fitness[best_idx])
    best_metrics = dict(evaluated[best_idx][2])
    best_candidate = evaluated[best_idx][1]
    no_improvement = 0
    stopped_early = False
    history = [_history_row(
        0, best_fitness, best_metrics, evaluated, best_initial_validation_accuracy,
        step0_source_index=step0_source_index,
        step0_source_scorecard_id=step0_source_id,
        step0_matches_best_initial_index=step0_matches_anchor_index,
        step0_matches_best_initial_id=step0_matches_anchor_id,
        step0_reconstruction_metrics=step0_reconstruction_metrics,
    )]
    if config.deterministic and config.checkpoint_interval > 0:
        _save_reproducibility_checkpoint(
            output_dir, 0, population, fitness, best_z, best_fitness, no_improvement, rng
        )

    if anchor_count == 0:
        step0_source_label = "random_latente"
    elif step0_matches_anchor_index:
        step0_source_label = "decoded_S1"
    else:
        step0_source_label = "vicino_anchor"

    print(
        "GA step 0 | "
        f"fitness={best_fitness:.4f} | "
        f"F1_val={best_metrics['validation_f1']:.4f} | "
        f"rules={int(best_metrics['n_scorecard_rules'])}"
    )
    if anchor_count > 0 and initial_metrics is not None and decoded_s1_metrics is not None:
        print(
            "GA step 0 confronto | "
            f"S1_original(fit={initial_fitness:.4f}, f1={initial_metrics['validation_f1']:.4f}, "
            f"rules={int(initial_metrics['n_scorecard_rules'])}) | "
            f"decoded_S1(fit={decoded_s1_fitness:.4f}, f1={decoded_s1_metrics['validation_f1']:.4f}, "
            f"rules={int(decoded_s1_metrics['n_scorecard_rules'])}, "
            f"delta_rules={int(decoded_s1_metrics['n_scorecard_rules']) - reference_n_rules:+d}, "
            f"decoded_prefix_rules={decoded_s1_metrics.get('decoded_n_rules_prefix')}) | "
            f"S2(fit={best_fitness:.4f}, f1={best_metrics['validation_f1']:.4f}, "
            f"rules={int(best_metrics['n_scorecard_rules'])}, "
            f"delta_rules={int(best_metrics['n_scorecard_rules']) - reference_n_rules:+d}, "
            f"decoded_prefix_rules={best_metrics.get('decoded_n_rules_prefix')})"
        )

    progress_path = output_dir / "ga_progress.json"

    def save_live_progress(step: int, stale: int, *, status: str) -> None:
        payload = {
            "status": status,
            "step": int(step),
            "total_steps": int(config.steps),
            "no_improvement_steps": int(stale),
            "patience": None if config.patience is None else int(config.patience),
            "best_fitness": float(best_fitness),
            "best_validation_f1": float(best_metrics["validation_f1"]),
            "best_n_rules": int(best_metrics["n_scorecard_rules"]),
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(progress_path)

    save_live_progress(0, no_improvement, status="running")
    progress = trange(
        1, config.steps + 1, desc="Ricerca genetica", unit="step", leave=True,
        dynamic_ncols=True,
    )
    for step in progress:
        order = _deterministic_order(fitness, evaluated, config)
        elite_indices = order[:config.elite_size]
        elites = population[elite_indices].copy()
        # Gli elite vengono copiati bit-a-bit nella generazione successiva:
        # fitness e ScoreCard sono quindi invariati e non vanno decodificati
        # né valutati di nuovo. Si valuta soltanto la progenie mutata.
        elite_fitness = fitness[elite_indices].copy()
        elite_evaluated = [evaluated[int(index)] for index in elite_indices]
        next_population = [candidate.copy() for candidate in elites]
        while len(next_population) < config.population_size:
            parent_a, parent_b = elites[rng.integers(config.elite_size, size=2)]
            alpha = rng.uniform(config.crossover_low, config.crossover_high)
            child = alpha * parent_a + (1.0 - alpha) * parent_b
            mutation_scale = config.mutation_std * (config.mutation_decay ** step)
            child += rng.normal(0.0, mutation_scale, size=latent_dim)
            next_population.append(child.astype(np.float32))

        population = np.asarray(next_population, dtype=np.float32)
        offspring_fitness, offspring_evaluated = _evaluate(
            population[len(elites):], decoder, data, config, validation_folds, reference_n_rules,
            fairness_context,
        )
        fitness = np.concatenate([elite_fitness, offspring_fitness])
        evaluated = elite_evaluated + offspring_evaluated
        if not np.isfinite(fitness).any():
            raise RuntimeError(f"Nessuna scorecard valida allo step {step} del GA.")
        if save_population_history:
            population_history.append(population.copy())
            fitness_history.append(fitness.copy())

        step_idx = int(_deterministic_order(fitness, evaluated, config)[0])
        step_best = float(fitness[step_idx])
        step_metrics = evaluated[step_idx][2]
        if step_best > best_fitness + config.min_delta:
            best_fitness = step_best
            best_z = population[step_idx].copy()
            best_metrics = dict(step_metrics)
            best_candidate = evaluated[step_idx][1]
            no_improvement = 0
            progress.write(
                f"NUOVO BEST step={step} | fitness={best_fitness:.4f} | "
                f"f1={best_metrics['validation_f1']:.4f} | "
                f"rules={int(best_metrics['n_scorecard_rules'])}"
            )
        else:
            no_improvement += 1

        history.append(_history_row(
            step,
            best_fitness,
            best_metrics,
            evaluated,
            best_initial_validation_accuracy,
            no_improvement=no_improvement,
            step_metrics=step_metrics,
            step_best_fitness=step_best,
        ))
        save_live_progress(step, no_improvement, status="running")
        if (
            config.deterministic
            and config.checkpoint_interval > 0
            and step % config.checkpoint_interval == 0
        ):
            _save_reproducibility_checkpoint(
                output_dir, step, population, fitness, best_z, best_fitness,
                no_improvement, rng,
            )
        progress.set_postfix(
            stale=no_improvement,
        )

        if config.patience is not None and no_improvement >= config.patience:
            stopped_early = True
            history[-1]["stopped_early"] = True
            progress.write(
                f"GA early stopping: step={step}, patience={config.patience}, "
                f"best fitness={best_fitness:.4f}, val_f1={best_metrics['validation_f1']:.4f}"
            )
            break

    progress.close()
    save_live_progress(
        int(history[-1]["step"]), no_improvement, status="completed"
    )
    _, candidate, final_metrics = _fitness(
        best_z, decoder, data, config, validation_folds, reference_n_rules,
        fairness_context,
    )
    if candidate is None:
        raise RuntimeError("La migliore soluzione genetica non e decodificabile.")
    if (
        config.f1_complexity_rule_objective == "target_constraint"
        and config.f1_complexity_f1_weight < 1.0 - 1e-12
    ):
        minimum_target = _requested_rule_target(
            reference_n_rules, config.f1_complexity_f1_weight
        )
        final_rules = int(final_metrics["n_scorecard_rules"])
        if final_rules < minimum_target:
            raise MinimumRuleTargetNotReached(
                f"Nessun candidato con almeno {minimum_target} regole; "
                f"migliore_sotto_target={final_rules}."
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "genetic_history.csv", index=False)
    if save_population_history:
        np.savez_compressed(
            output_dir / "population_history_alpha_0.7.npz",
            steps=history_df["step"].to_numpy(dtype=np.int64),
            populations=np.asarray(population_history, dtype=np.float32),
            fitness=np.asarray(fitness_history, dtype=np.float64),
        )
    np.save(output_dir / "best_genetic_latent_vector.npy", best_z)
    candidate.rows.to_csv(output_dir / "best_genetic_scorecard_rows.csv", index=False)
    candidate.leaves.to_csv(output_dir / "best_genetic_scorecard_leaves.csv", index=False)
    pd.DataFrame([final_metrics]).to_csv(output_dir / "best_genetic_fitness_metrics.csv", index=False)

    # Artefatti opzionali per il raffinamento VAE-GA. La popolazione qui è
    # esattamente quella dell'ultimo step eseguito (anche in caso di early
    # stopping) e ``evaluated`` contiene già tutte le relative decodifiche:
    # non viene quindi effettuata una seconda valutazione né usato il test set.
    if os.environ.get("GENRULECARD_SAVE_FINAL_POPULATION") == "1":
        np.save(output_dir / "final_population_latent.npy", population)
        metric_rows = []
        leaves_frames = []
        tree_frames = []
        scorecard_frames = []
        for population_index, (fit_value, evaluated_item) in enumerate(
            zip(fitness, evaluated)
        ):
            _, decoded, metrics = evaluated_item
            row = {
                "population_index": int(population_index),
                "valid": decoded is not None,
                "selection_fitness": float(fit_value),
            }
            for key, value in metrics.items():
                if np.isscalar(value) or value is None:
                    row[key] = value
            metric_rows.append(row)
            if decoded is None:
                continue
            for frame, destination in (
                (decoded.leaves, leaves_frames),
                (decoded.rows, tree_frames),
            ):
                saved = frame.copy()
                saved.insert(0, "population_index", int(population_index))
                destination.append(saved)
            scorecard_frames.append(pd.DataFrame([
                {
                    "population_index": int(population_index),
                    "rule_index": int(rule_index),
                    "rule_repr": repr(rule),
                }
                for rule_index, rule in enumerate(decoded.scorecard.scorecard)
            ]))
        pd.DataFrame(metric_rows).to_csv(
            output_dir / "final_population_metrics.csv", index=False
        )
        for filename, frames in (
            ("final_population_leaves.csv", leaves_frames),
            ("final_population_tree_rows.csv", tree_frames),
            ("final_population_scorecard_rules.csv", scorecard_frames),
        ):
            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            combined.to_csv(output_dir / filename, index=False)
        (output_dir / "final_population_manifest.json").write_text(json.dumps({
            "population_size": int(len(population)),
            "valid_candidates": int(sum(row["valid"] for row in metric_rows)),
            "last_step": int(history_df["step"].iloc[-1]),
            "stopped_early": bool(stopped_early),
            "scope": "last GA population; validation metrics only",
        }, indent=2), encoding="utf-8")

    candidate_info = dict(candidate.info)
    candidate_info.update({
        "selection_fitness": final_metrics["selection_fitness"],
        "validation_accuracy": final_metrics["validation_accuracy"],
        "validation_f1": final_metrics["validation_f1"],
        "fold_performance_mean": final_metrics["fold_performance_mean"],
        "fold_performance_std": final_metrics["fold_performance_std"],
        "fold_performance_min": final_metrics["fold_performance_min"],
        "robust_validation_performance": final_metrics["robust_validation_performance"],
        "confidence_tiebreak": final_metrics["confidence_tiebreak"],
        "complexity_penalty": final_metrics["complexity_penalty"],
        "reference_n_scorecard_rules": final_metrics["reference_n_scorecard_rules"],
        "redundancy_penalty": final_metrics["redundancy_penalty"],
        "fitness_metric": config.fitness_metric,
        "step0_source_index": step0_source_index,
        "step0_source_scorecard_id": step0_source_id,
        "step0_matches_best_initial_index": step0_matches_anchor_index,
        "step0_matches_best_initial_id": step0_matches_anchor_id,
        "step0_best_ga_is_decoded_best_initial": step0_matches_anchor_index,
    })
    for key in (
        "fairness_objective", "demographic_parity_difference",
        "equal_opportunity_difference", "equalized_odds_difference",
        "fairness_dpd_violation", "fairness_eodd_violation",
        "fairness_penalty", "fairness_component", "fairness_score",
        "simplicity_score", "fairness_eligible",
    ):
        if key in final_metrics:
            candidate_info[key] = final_metrics[key]
    candidate_info.update(step0_reconstruction_metrics)
    status = "early stopping" if stopped_early else "completato"
    if os.environ.get("GENRULECARD_CLEAN_LOGS") != "1":
        print(
            f"GA {status} | best val_f1={final_metrics['validation_f1']:.4f} | "
            f"fitness={final_metrics['selection_fitness']:.4f}"
        )
    return GeneticResult(
        best_z=best_z,
        validation_accuracy=final_metrics["validation_accuracy"],
        rows=candidate.rows,
        leaves=candidate.leaves,
        prior_log_odds=candidate.prior_log_odds,
        scorecard=candidate.scorecard,
        history=history_df,
        info=candidate_info,
    )
