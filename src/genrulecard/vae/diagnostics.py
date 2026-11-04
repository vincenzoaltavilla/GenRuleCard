from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

from ..core.artifacts import CorpusBundle, ProcessedDataset
from ..core.config import RuleCardConfig
from ..rulecard_fit.scorecards import import_rulecard, scorecard_from_leaves, scorecard_outputs


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> float:
    true_pos = np.asarray(y_true) == positive_label
    pred_pos = np.asarray(y_pred) == positive_label
    tp = float(np.logical_and(true_pos, pred_pos).sum())
    fp = float(np.logical_and(~true_pos, pred_pos).sum())
    fn = float(np.logical_and(true_pos, ~pred_pos).sum())
    denominator = 2.0 * tp + fp + fn
    return 0.0 if denominator == 0.0 else 2.0 * tp / denominator


def _branch_role(operator) -> str:
    return "L" if str(operator) in ("<", "<=") else "R"


def _positioned_leaves(leaves: pd.DataFrame) -> dict[tuple[int, str], tuple]:
    positioned = {}
    for _, row in leaves.iterrows():
        path = row.get("conditions_key")
        if path is None or len(path) == 0:
            continue
        role = "".join(_branch_role(condition[2]) for condition in path)
        positioned[(int(row["tree_idx"]), role)] = tuple(path)
    return positioned


def _feature_operator_signature(path: tuple) -> tuple:
    return tuple((int(condition[0]), str(condition[2])) for condition in path)


def _positional_reconstruction_metrics(
    original: pd.DataFrame,
    decoded: pd.DataFrame,
    feature_range: np.ndarray,
) -> tuple[int, float, float]:
    original_by_position = _positioned_leaves(original)
    decoded_by_position = _positioned_leaves(decoded)
    positions = set(original_by_position) | set(decoded_by_position)
    if not positions:
        return 0, np.nan, np.nan
    signature_delta = 0
    matched = 0
    threshold_errors = []
    for position in positions:
        if position not in original_by_position or position not in decoded_by_position:
            signature_delta += 1
            continue
        original_path = original_by_position[position]
        decoded_path = decoded_by_position[position]
        if _feature_operator_signature(original_path) != _feature_operator_signature(decoded_path):
            signature_delta += 2
            continue
        matched += 1
        for original_condition, decoded_condition in zip(original_path, decoded_path):
            feature = int(original_condition[0])
            scale = max(abs(float(feature_range[feature])), 1e-8)
            threshold_errors.append(
                abs(float(decoded_condition[1]) - float(original_condition[1]))
                / scale * 100.0
            )
    threshold_error = float(np.mean(threshold_errors)) if threshold_errors else np.nan
    return int(signature_delta), threshold_error, float(matched / len(positions))


def diagnose_reconstruction(
    embeddings: np.ndarray,
    decoder,
    corpus: CorpusBundle,
    data: ProcessedDataset,
    output_dir: Path,
) -> pd.DataFrame:
    """Audit natural decoder sulle stesse 300 ScoreCard del corpus."""
    cards = corpus.cards_df.set_index("scorecard_id")
    rows = []
    failures = []
    original_exports = []
    decoded_exports = []
    for position, scorecard_id in enumerate(corpus.card_ids):
        original = corpus.corpus_df.loc[
            corpus.corpus_df["scorecard_id"] == scorecard_id
        ].copy()
        try:
            decoded = decoder.decode(
                np.asarray(embeddings[position], dtype=np.float32),
                scorecard_id=10_000_000 + int(scorecard_id),
            )
            _, prediction = scorecard_outputs(decoded.scorecard, data.X_validation)
            decoded_f1 = _binary_f1(
                data.y_validation, prediction, data.model_classes[-1]
            )
            original_f1 = float(cards.loc[scorecard_id, "validation_f1"])
            signature_delta, threshold_error, coverage = _positional_reconstruction_metrics(
                original, decoded.leaves, data.feature_range
            )
            original_export = original.copy()
            original_export.insert(0, "source_scorecard_id", int(scorecard_id))
            decoded_export = decoded.leaves.copy()
            decoded_export.insert(0, "source_scorecard_id", int(scorecard_id))
            original_exports.append(original_export)
            decoded_exports.append(decoded_export)
            rows.append({
                "scorecard_id": int(scorecard_id),
                "original_f1": original_f1,
                "decoded_f1": decoded_f1,
                "delta_f1": decoded_f1 - original_f1,
                "original_n_leaves": int(len(original)),
                "decoded_n_leaves": int(len(decoded.leaves)),
                "delta_n_leaves": int(len(decoded.leaves) - len(original)),
                "delta_leaf_count_feat_op_abs_sum": signature_delta,
                "mean_abs_pct_delta_threshold": threshold_error,
                "threshold_match_coverage": coverage,
            })
        except Exception as error:
            failures.append({
                "scorecard_id": int(scorecard_id),
                "error_type": type(error).__name__,
                "error": str(error),
            })
    report = pd.DataFrame(rows)
    report.to_csv(output_dir / "decoded_reconstruction_fmi_f1_sig.csv", index=False)
    original_export_frame = (
        pd.concat(original_exports, ignore_index=True)
        if original_exports else pd.DataFrame()
    )
    decoded_export_frame = (
        pd.concat(decoded_exports, ignore_index=True)
        if decoded_exports else pd.DataFrame()
    )
    original_export_frame.to_csv(
        output_dir / "reconstruction_original_leaves.csv", index=False
    )
    decoded_export_frame.to_csv(
        output_dir / "reconstruction_decoded_leaves.csv", index=False
    )
    pd.DataFrame(failures).to_csv(
        output_dir / "decoded_reconstruction_failures_fmi_f1_sig.csv", index=False
    )
    return report


def _knn_overlap(distance_a: np.ndarray, distance_b: np.ndarray, k: int) -> float:
    overlap = []
    for index in range(len(distance_a)):
        a = set(np.argsort(distance_a[index])[1:k + 1])
        b = set(np.argsort(distance_b[index])[1:k + 1])
        overlap.append(len(a & b) / k)
    return float(np.mean(overlap))


def _positive_probabilities(scorecard, X: np.ndarray) -> np.ndarray:
    probabilities, _ = scorecard_outputs(scorecard, X)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim == 1:
        return probabilities
    if probabilities.ndim == 2:
        return probabilities[:, -1]
    raise ValueError(f"Output inatteso da ScoreCard.predict_proba: {probabilities.shape}")


def _summary(name: str, reference: np.ndarray, latent: np.ndarray, upper: tuple[np.ndarray, np.ndarray]) -> dict:
    ref_pairs, lat_pairs = reference[upper], latent[upper]
    rho, rho_p = spearmanr(ref_pairs, lat_pairs)
    pearson, pearson_p = pearsonr(ref_pairs, lat_pairs)
    near = np.quantile(ref_pairs, .10)
    far = np.quantile(ref_pairs, .90)
    k = min(10, len(reference) - 1)
    mean_near = float(lat_pairs[ref_pairs <= near].mean())
    mean_far = float(lat_pairs[ref_pairs >= far].mean())
    return {
        "distance_type": name,
        "n_pairs": int(len(ref_pairs)),
        "spearman_rho": float(rho),
        "spearman_pvalue": float(rho_p),
        "pearson_r": float(pearson),
        "pearson_pvalue": float(pearson_p),
        "mean_latent_distance_near_q10": mean_near,
        "mean_latent_distance_far_q90": mean_far,
        "near_far_latent_ratio": mean_near / max(mean_far, 1e-8),
        "knn_overlap_at_10": _knn_overlap(reference, latent, k),
    }


def diagnose_representation(
    embeddings: np.ndarray,
    structural_distance: np.ndarray,
    corpus: CorpusBundle,
    data: ProcessedDataset,
    rulecard_config: RuleCardConfig,
    source_dir: Path | str | None,
    output_dir: Path,
    seed: int,
    diagnostic_indices: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    _, _, scorecard_cls = import_rulecard(source_dir)
    cards_by_id = corpus.cards_df.set_index("scorecard_id")
    probabilities = []
    for identifier in corpus.card_ids:
        leaves = corpus.corpus_df.loc[corpus.corpus_df["scorecard_id"] == identifier].copy()
        prior = float(cards_by_id.loc[identifier, "prior_log_odds"])
        scorecard = scorecard_from_leaves(leaves, prior, data.model_classes, rulecard_config, scorecard_cls)
        probabilities.append(_positive_probabilities(scorecard, data.X_train))
    predictive_probabilities = np.vstack(probabilities).astype(np.float32)
    predictive_distance = (cdist(predictive_probabilities, predictive_probabilities) / np.sqrt(predictive_probabilities.shape[1])).astype(np.float32)
    np.fill_diagonal(predictive_distance, 0.0)
    latent_distance = cdist(embeddings, embeddings).astype(np.float32)
    np.fill_diagonal(latent_distance, 0.0)
    if diagnostic_indices is None:
        report_structural = structural_distance
        report_latent = latent_distance
        report_predictive = predictive_distance
        report_scope = "all"
    else:
        diagnostic_indices = np.asarray(diagnostic_indices, dtype=np.int64)
        selection = np.ix_(diagnostic_indices, diagnostic_indices)
        report_structural = structural_distance[selection]
        report_latent = latent_distance[selection]
        report_predictive = predictive_distance[selection]
        report_scope = "holdout"
    upper = np.triu_indices(len(report_structural), k=1)
    report = pd.DataFrame([
        _summary("structural_pi_T", report_structural, report_latent, upper),
        _summary("predictive_probability_on_train", report_predictive, report_latent, upper),
    ])
    report.insert(0, "scope", report_scope)
    report.insert(1, "n_scorecards", len(report_structural))
    pca_array = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    validation_accuracies = cards_by_id.loc[corpus.card_ids, "validation_accuracy"].to_numpy()
    pca_df = pd.DataFrame({
        "scorecard_id": corpus.card_ids, "pc1": pca_array[:, 0], "pc2": pca_array[:, 1],
        "validation_accuracy": validation_accuracies,
    })
    np.save(output_dir / "latent_embeddings.npy", embeddings)
    np.save(output_dir / "latent_distance_matrix.npy", latent_distance)
    np.save(output_dir / "structural_distance_matrix.npy", structural_distance)
    np.save(output_dir / "predictive_probabilities_on_train.npy", predictive_probabilities)
    np.save(output_dir / "predictive_distance_matrix.npy", predictive_distance)
    report.to_csv(output_dir / "representation_distance_diagnostics.csv", index=False)
    pca_df.to_csv(output_dir / "latent_pca.csv", index=False)
    structural = report.loc[report["distance_type"] == "structural_pi_T"].iloc[0]
    predictive = report.loc[report["distance_type"] == "predictive_probability_on_train"].iloc[0]
    print(f"VAE diagnostics | scope={report_scope} | n={len(report_structural)}")
    print(f"Structural | Spearman={structural['spearman_rho']:.4f} | Pearson={structural['pearson_r']:.4f} | kNN@10={structural['knn_overlap_at_10']:.4f}")
    return report, pca_df, latent_distance, predictive_distance
