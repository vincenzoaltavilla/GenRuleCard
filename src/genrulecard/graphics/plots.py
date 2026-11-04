from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_phase_timing_plot(timings: pd.DataFrame, output_dir: Path) -> None:
    """Salva il riepilogo conclusivo dei tempi di tutte le fasi."""
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    colors = [palette[index % len(palette)] for index in range(len(timings))]
    bars = ax.bar(timings["phase_label"], timings["seconds"], color=colors)
    ax.bar_label(
        bars,
        labels=[f"{seconds:.1f}s\n({seconds / 60.0:.1f} min)" for seconds in timings["seconds"]],
        padding=4,
        fontsize=9,
    )
    ax.set(
        title="Tempi delle fasi GenRuleCard",
        xlabel="Fase",
        ylabel="Tempo (secondi)",
    )
    ax.grid(axis="y", alpha=.25)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(output_dir / "genrulecard_phase_timings.png", dpi=180)
    plt.close(fig)


def save_vae_training_plots(early_history: pd.DataFrame, final_history: pd.DataFrame, output_dir: Path) -> None:
    if not early_history.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(early_history["epoch"], early_history["vae_train_total"], label="VAE train interno")
        ax.plot(early_history["epoch"], early_history["vae_earlystop_total"], label="VAE holdout interno")
        monitor_column = "vae_earlystop_total"
        if "vae_earlystop_monitored" in early_history.columns:
            monitored = early_history["vae_earlystop_monitored"].to_numpy(dtype=float)
            holdout_total = early_history["vae_earlystop_total"].to_numpy(dtype=float)
            if not np.allclose(monitored, holdout_total, equal_nan=True):
                monitor_column = "vae_earlystop_monitored"
        best_epoch = int(early_history.loc[early_history[monitor_column].idxmin(), "epoch"])
        monitor_label = "metrica monitorata" if monitor_column == "vae_earlystop_monitored" else "loss totale"
        ax.axvline(best_epoch, linestyle="--", label=f"best={best_epoch} ({monitor_label})")
        ax.set(title="Scelta epoca VAE", xlabel="Epoca", ylabel="Loss totale")
        ax.grid(alpha=.25)
        if monitor_column != "vae_earlystop_total":
            monitor_ax = ax.twinx()
            monitor_ax.plot(
                early_history["epoch"], early_history[monitor_column],
                color="tab:green", alpha=.75, linestyle=":", label="Metrica monitorata",
            )
            monitor_ax.set_ylabel("Loss metrica")
            lines, labels = ax.get_legend_handles_labels()
            monitor_lines, monitor_labels = monitor_ax.get_legend_handles_labels()
            ax.legend(lines + monitor_lines, labels + monitor_labels)
        else:
            ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "vae_earlystop_curves.png", dpi=180)
        plt.close(fig)
    if not final_history.empty:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(final_history["epoch"], final_history["full_T_total"], label="VAE finale su tutto T")
        ax.set(title="Refit VAE su tutto il corpus T", xlabel="Epoca", ylabel="Loss totale")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "vae_refit_curve.png", dpi=180)
        plt.close(fig)


def save_representation_plots(
    report: pd.DataFrame,
    structural_distance: np.ndarray,
    predictive_distance: np.ndarray,
    latent_distance: np.ndarray,
    latent_pca: pd.DataFrame,
    output_dir: Path,
    seed: int,
    comparison_marker: dict[str, float | str] | None = None,
) -> None:
    upper = np.triu_indices(len(structural_distance), k=1)
    x, y = structural_distance[upper], latent_distance[upper]
    rng = np.random.default_rng(seed)
    if len(x) > 18000:
        idx = rng.choice(len(x), size=18000, replace=False)
        x, y = x[idx], y[idx]
    structural = report.loc[report["distance_type"] == "structural_pi_T"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=8, alpha=.18)
    if comparison_marker is not None:
        ax.scatter(
            [float(comparison_marker["structural_distance"])],
            [float(comparison_marker["latent_distance"])],
            s=90,
            marker="X",
            color="#d62728",
            edgecolor="black",
            linewidth=0.8,
            label=str(comparison_marker.get("label", "best initial -> best genetic")),
        )
        ax.legend()
    ax.set(
        title=f"Struttura → spazio latente\nSpearman ρ={structural['spearman_rho']:.3f} | Pearson r={structural['pearson_r']:.3f}",
        xlabel="Distanza strutturale π(T)", ylabel="Distanza euclidea latente",
    )
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "structural_vs_latent_distance.png", dpi=180)
    plt.close(fig)

    upper = np.triu_indices(len(predictive_distance), k=1)
    x, y = predictive_distance[upper], latent_distance[upper]
    if len(x) > 18000:
        idx = rng.choice(len(x), size=18000, replace=False)
        x, y = x[idx], y[idx]
    predictive = report.loc[report["distance_type"] == "predictive_probability_on_train"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=8, alpha=.18)
    if comparison_marker is not None:
        ax.scatter(
            [float(comparison_marker["predictive_distance"])],
            [float(comparison_marker["latent_distance"])],
            s=90,
            marker="X",
            color="#d62728",
            edgecolor="black",
            linewidth=0.8,
            label=str(comparison_marker.get("label", "best initial -> best genetic")),
        )
        ax.legend()
    ax.set(
        title=f"Predizione -> spazio latente\nSpearman rho={predictive['spearman_rho']:.3f} | Pearson r={predictive['pearson_r']:.3f}",
        xlabel="Distanza predittiva su train", ylabel="Distanza euclidea latente",
    )
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "predictive_vs_latent_distance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(latent_pca["pc1"], latent_pca["pc2"], c=latent_pca["validation_accuracy"], s=28, alpha=.85)
    ax.set(title="PCA degli embedding VAE", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=.25)
    fig.colorbar(scatter, ax=ax, label="Accuracy validation")
    fig.tight_layout()
    fig.savefig(output_dir / "latent_pca_validation_accuracy.png", dpi=180)
    plt.close(fig)


def save_ga_plot(history: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history["step"], history["best_validation_f1"], label="Best GA F1")
    ax.plot(
        history["step"], history["mean_validation_f1"],
        alpha=.70, label="Media popolazione F1",
    )
    initial_column = (
        "step0_s1_original_f1"
        if "step0_s1_original_f1" in history.columns
        else "best_validation_f1"
    )
    initial_values = history[initial_column].dropna()
    if not initial_values.empty:
        ax.axhline(
            float(initial_values.iloc[0]), color="black", linestyle="--",
            label="ScoreCard iniziale F1",
        )
    ax.set(
        title="Evoluzione F1 del GA nello spazio latente",
        xlabel="Step genetico", ylabel="F1 validation",
    )
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "genetic_convergence.png", dpi=180)
    plt.close(fig)


def save_vae_reconstruction_8plots(
    reconstruction: pd.DataFrame,
    fmi_distance: np.ndarray,
    latent_distance: np.ndarray,
    dataset_name: str,
    output_dir: Path,
) -> None:
    """Report VAE coerente con gli otto pannelli del notebook benchmark."""
    if reconstruction.empty:
        return
    upper = np.triu_indices(len(fmi_distance), k=1)
    fmi_pairs = fmi_distance[upper]
    latent_pairs = latent_distance[upper]
    pearson = float(np.corrcoef(fmi_pairs, latent_pairs)[0, 1])
    spearman = float(pd.Series(fmi_pairs).corr(pd.Series(latent_pairs), method="spearman"))

    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    fig.suptitle(f"{dataset_name} | VAE_fmi_f1_sig", fontsize=18)
    g = reconstruction

    ax = axes[0, 0]
    ax.scatter(g["original_f1"], g["decoded_f1"], alpha=.55, s=18)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlim=(0, 1), ylim=(0, 1), title="F1 originale vs decoded",
           xlabel="original validation F1", ylabel="decoded validation F1")

    ax = axes[0, 1]
    ax.hist(g["delta_f1"], bins=35, alpha=.85)
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(-.10, color="red", linestyle=":")
    ax.axvline(.10, color="red", linestyle=":")
    ax.set_title("delta_f1")

    ax = axes[1, 0]
    ax.scatter(g["original_n_leaves"], g["decoded_n_leaves"], alpha=.55, s=18)
    limit = max(g["original_n_leaves"].max(), g["decoded_n_leaves"].max()) + 2
    ax.plot([0, limit], [0, limit], "k--")
    ax.set(title="n_leaves originale vs decoded", xlabel="original n_leaves",
           ylabel="decoded n_leaves")

    ax = axes[1, 1]
    ax.hist(g["delta_n_leaves"], bins=35, alpha=.85)
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(-10, color="red", linestyle=":")
    ax.axvline(10, color="red", linestyle=":")
    ax.set_title("delta_n_leaves")

    ax = axes[2, 0]
    ax.hist(g["delta_leaf_count_feat_op_abs_sum"], bins=35, alpha=.85)
    ax.axvline(0, color="black", linestyle="--")
    ax.set(title="delta_leaf_count_feat_op_abs_sum",
           xlabel="somma |delta conteggi feature-op|")

    ax = axes[2, 1]
    threshold = g["mean_abs_pct_delta_threshold"].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    ax.hist(threshold, bins=35, alpha=.85)
    ax.axvline(0, color="black", linestyle="--")
    ax.axvline(5, color="red", linestyle=":")
    ax.axvline(10, color="red", linestyle=":")
    mean_coverage = float(g["threshold_match_coverage"].mean())
    ax.set(
        title=f"threshold error su match posizionali | mean coverage={mean_coverage:.1%}",
        xlabel="errore medio assoluto / feature range (%)",
    )

    ax = axes[3, 0]
    ax.scatter(fmi_pairs, latent_pairs, alpha=.15, s=8)
    ax.set(
        title=f"D_FMI vs distanza latente | Pearson={pearson:.3f}, Spearman={spearman:.3f}",
        xlabel="D_FMI = 1 - FMI", ylabel="distanza euclidea latente",
    )

    ax = axes[3, 1]
    ks = [k for k in (1, 3, 5, 10, 25, 50, 100) if k < len(fmi_distance)]
    overlaps = []
    for k in ks:
        values = []
        for row in range(len(fmi_distance)):
            fmi_neighbors = np.argsort(fmi_distance[row])[1:k + 1]
            latent_neighbors = np.argsort(latent_distance[row])[1:k + 1]
            values.append(len(set(fmi_neighbors) & set(latent_neighbors)) / k)
        overlaps.append(float(np.mean(values)))
    ax.plot(ks, overlaps, marker="o")
    ax.set(ylim=(0, 1), title="Overlap vicini FMI vs vicini latenti",
           xlabel="k", ylabel="mean overlap@k / k")

    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(output_dir / "vae_reconstruction_fmi_8plots.png", dpi=180)
    plt.close(fig)


def save_f1_complexity_weight_sweep_plots(
    results: pd.DataFrame,
    output_dir: Path,
    dataset: str | None = None,
) -> None:
    """Due grafici: trade-off regole/F1 su validation e su test."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, delta_column, filename in (
        ("validation", "delta_validation_f1", "f1_weight_sweep_validation.png"),
        ("test", "delta_test_f1", "f1_weight_sweep_test.png"),
    ):
        ordered = results.sort_values("f1_weight")
        fig, left = plt.subplots(figsize=(9, 5))
        right = left.twinx()
        f1_line = left.plot(
            ordered["f1_weight"], ordered[delta_column],
            color="tab:blue", marker="o", label=f"ΔF1 {split}",
        )
        rules_line = right.plot(
            ordered["f1_weight"], ordered["delta_n_rules"],
            color="tab:red", marker="s", label="Δn_rules",
        )
        left.axhline(0, color="tab:blue", linestyle=":", alpha=.55)
        right.axhline(0, color="tab:red", linestyle=":", alpha=.55)
        left.set(
            xlim=(0, 1), xticks=np.linspace(0, 1, 11),
            xlabel="f1_complexity_f1_weight",
            ylabel=f"ΔF1 {split}",
            title=(
                f"{dataset.capitalize()} | trade-off F1-complexity ({split})"
                if dataset else f"Trade-off F1-complexity ({split})"
            ),
        )
        right.set_ylabel("Δ numero di regole")
        left.grid(alpha=.25)
        lines = f1_line + rules_line
        left.legend(lines, [line.get_label() for line in lines], loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

# ============================================================================
# Visualizzazione RuleCard rosso/verde del notebook
# ============================================================================

def scorecard_rules_to_df(
    scorecard: Any,
    feature_names: list[str],
    *,
    aggregate_duplicates: bool = True,
    onehot_mapping: dict[str, dict] | None = None,
    boolean_features: list[str] | None = None,
    onehot_threshold: float = 0.5,
    drop_zero_points: bool = True,
) -> pd.DataFrame:
    """Converte ``scorecard.scorecard`` in regole leggibili per il plot.

    Mantiene la semantica ufficiale della classe ``ScoreCard``: punti arrotondati,
    aggregazione di duplicati ed esclusione delle regole con score finale zero.
    """
    if onehot_mapping is None:
        onehot_mapping = {}
    boolean_features = set(boolean_features or [])

    def format_category_value(value: Any) -> str:
        if isinstance(value, str):
            return value if value in ("True", "False") else f"'{value}'"
        return str(value)

    def format_onehot_condition(name: str, lower, lower_op, upper, upper_op) -> str:
        info = onehot_mapping[name]
        original_feature = info["original_feature"]
        category = info["category"]
        all_categories = info.get("all_categories")

        if lower is not None and upper is None:
            if abs(lower - onehot_threshold) < 1e-9 and lower_op in (">", ">="):
                return f"{original_feature} = {format_category_value(category)}"
        if upper is not None and lower is None:
            if abs(upper - onehot_threshold) < 1e-9 and upper_op in ("<", "<="):
                if all_categories is not None and len(all_categories) == 2:
                    other = [value for value in all_categories if str(value) != str(category)]
                    if len(other) == 1:
                        return f"{original_feature} = {format_category_value(other[0])}"
                return f"{original_feature} != {format_category_value(category)}"

        parts: list[str] = []
        if lower is not None:
            parts.append(f"{name} {lower_op} {lower:.2f}")
        if upper is not None:
            parts.append(f"{name} {upper_op} {upper:.2f}")
        return " AND ".join(parts)

    def format_boolean_condition(name: str, lower, lower_op, upper, upper_op) -> str:
        if lower is not None and upper is None:
            if abs(lower - onehot_threshold) < 1e-9 and lower_op in (">", ">="):
                return f"{name} = True"
        if upper is not None and lower is None:
            if abs(upper - onehot_threshold) < 1e-9 and upper_op in ("<", "<="):
                return f"{name} = False"

        parts: list[str] = []
        if lower is not None:
            parts.append(f"{name} {lower_op} {lower:.2f}")
        if upper is not None:
            parts.append(f"{name} {upper_op} {upper:.2f}")
        return " AND ".join(parts)

    rows: list[dict[str, Any]] = []
    for _, conditions, support, points in scorecard.scorecard:
        feature_bounds: dict[int, dict[str, Any]] = {}
        for feature_idx, threshold, operator in conditions:
            feature_idx = int(feature_idx)
            threshold = float(threshold)
            feature_bounds.setdefault(feature_idx, {"lower": None, "lower_op": None, "upper": None, "upper_op": None})
            if operator in (">", ">="):
                if feature_bounds[feature_idx]["lower"] is None or threshold > feature_bounds[feature_idx]["lower"]:
                    feature_bounds[feature_idx]["lower"] = threshold
                    feature_bounds[feature_idx]["lower_op"] = operator
            elif operator in ("<", "<="):
                if feature_bounds[feature_idx]["upper"] is None or threshold < feature_bounds[feature_idx]["upper"]:
                    feature_bounds[feature_idx]["upper"] = threshold
                    feature_bounds[feature_idx]["upper_op"] = operator
            else:
                raise ValueError(f"Operatore non riconosciuto: {operator}")

        parts: list[str] = []
        key_parts: list[tuple] = []
        for feature_idx in sorted(feature_bounds):
            name = feature_names[feature_idx]
            lower = feature_bounds[feature_idx]["lower"]
            lower_op = feature_bounds[feature_idx]["lower_op"]
            upper = feature_bounds[feature_idx]["upper"]
            upper_op = feature_bounds[feature_idx]["upper_op"]
            key_parts.append((feature_idx, lower, lower_op, upper, upper_op))

            if name in onehot_mapping:
                parts.append(format_onehot_condition(name, lower, lower_op, upper, upper_op))
            elif name in boolean_features:
                parts.append(format_boolean_condition(name, lower, lower_op, upper, upper_op))
            elif lower is not None and upper is not None:
                left_operator = "<" if lower_op == ">" else "<="
                parts.append(f"{lower:.2f} {left_operator} {name} {upper_op} {upper:.2f}")
            elif lower is not None:
                parts.append(f"{name} {lower_op} {lower:.2f}")
            elif upper is not None:
                parts.append(f"{name} {upper_op} {upper:.2f}")

        rows.append({
            "rule": "  AND  ".join(parts),
            "points_rounded": int(points),
            "support": support,
            "conditions_key": tuple(key_parts),
        })

    rules_df = pd.DataFrame(rows)
    if rules_df.empty:
        return rules_df
    if not aggregate_duplicates:
        return (
            rules_df.loc[rules_df["points_rounded"] != 0]
            .sort_values("points_rounded", key=lambda series: series.abs(), ascending=False)
            .reset_index(drop=True)
        )

    rules_df["_conditions_key"] = rules_df["conditions_key"].map(repr)
    rules_df = (
        rules_df.groupby(["_conditions_key", "rule"], as_index=False)
        .agg(
            points_rounded=("points_rounded", "sum"),
            n_occurrences=("points_rounded", "size"),
            support_list=("support", list),
            conditions_key=("conditions_key", "first"),
        )
        .drop(columns="_conditions_key")
    )
    if drop_zero_points:
        rules_df = rules_df.loc[rules_df["points_rounded"] != 0].copy()
    return (
        rules_df.sort_values("points_rounded", key=lambda series: series.abs(), ascending=False)
        .reset_index(drop=True)
    )


def plot_rulecard_table(
    rules_df: pd.DataFrame,
    classes: list[Any] | tuple[Any, Any] | np.ndarray,
    *,
    rule_col: str = "rule",
    score_col: str = "points_rounded",
    max_rules: int | None = 25,
    figsize: tuple[float, float] = (7.0, 7.0),
    row_height: float = 0.30,
    cmap_colors: tuple[str, str, str] = ("#e60000", "#f7f7f7", "#008c3a"),
    other_color: str = "#bdbdbd",
    bar_label: str = "Class",
    text_white_threshold: float = 0.55,
    fontsize: float = 10,
    header_fontsize: float = 11,
    sort_by_score: bool = True,
    save_path: Path | str | None = None,
    show: bool = False,
):
    """Plot identico nella logica al grafico rosso/verde del notebook."""
    if rules_df.empty:
        raise ValueError("Nessuna regola con punti diversi da zero da visualizzare.")

    df = rules_df[[rule_col, score_col]].copy()
    df[score_col] = df[score_col].astype(float)
    if sort_by_score:
        df = df.sort_values(score_col, ascending=False)
    df = df.reset_index(drop=True)

    if max_rules is not None and len(df) > max_rules:
        if max_rules < 3:
            raise ValueError("max_rules deve essere almeno 3.")
        n_top = max_rules // 2
        n_bottom = max_rules - n_top - 1
        top_df = df.head(n_top).copy()
        bottom_df = df.tail(n_bottom).copy()
        hidden_df = df.iloc[n_top: len(df) - n_bottom].copy()
        other_row = pd.DataFrame({rule_col: [f"Other {len(hidden_df)} rules"], score_col: [hidden_df[score_col].sum()], "is_other": [True]})
        top_df["is_other"] = False
        bottom_df["is_other"] = False
        df = pd.concat([top_df, other_row, bottom_df], axis=0).reset_index(drop=True)
    else:
        df["is_other"] = False

    split_rules = df[rule_col].astype(str).str.split(r"\s+AND\s+", regex=True)
    max_conditions = max(split_rules.apply(len).max(), 2)
    condition_cols = [f"cond_{index + 1}" for index in range(max_conditions)]

    def align_conditions(parts, is_other: bool = False) -> list[str]:
        parts = list(parts)
        if is_other or len(parts) == 1:
            return [""] * (max_conditions - 1) + parts
        return parts + [""] * (max_conditions - len(parts))

    cond_df = pd.DataFrame(
        [align_conditions(parts, is_other=is_other) for parts, is_other in zip(split_rules, df["is_other"])],
        columns=condition_cols,
    )
    df_plot = pd.concat([cond_df, df[[score_col, "is_other"]]], axis=1)
    n_rows = len(df_plot)
    real_scores = df_plot.loc[~df_plot["is_other"], score_col]
    vmax = max(abs(real_scores.min()), abs(real_scores.max()))
    vmax = vmax if vmax > 0 else 1
    norm = mpl.colors.Normalize(vmin=-vmax, vmax=vmax)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("rulecard_cmap", list(cmap_colors), N=256)

    fig_height = max(figsize[1], row_height * (n_rows + 2))
    fig = plt.figure(figsize=(figsize[0], fig_height))
    ax = fig.add_axes([0.04, 0.045, 0.76, 0.91])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, n_rows + 1)
    ax.axis("off")
    cax = fig.add_axes([0.85, 0.23, 0.055, 0.54])

    score_width = 0.16
    rules_width = 1.0 - score_width
    ampersand_width = 0.04 if max_conditions > 1 else 0.0
    total_ampersand_width = ampersand_width * (max_conditions - 1)
    condition_width = (rules_width - total_ampersand_width) / max_conditions

    x_starts: list[float] = []
    current_x = 0.0
    for index in range(max_conditions):
        x_starts.append(current_x)
        current_x += condition_width
        if index < max_conditions - 1:
            current_x += ampersand_width

    score_x_start = rules_width
    header_y = n_rows + 0.35
    ax.text(rules_width / 2, header_y, "rule", ha="center", va="center", fontsize=header_fontsize, fontweight="bold")
    ax.text(score_x_start + score_width / 2, header_y, "score", ha="center", va="center", fontsize=header_fontsize, fontweight="bold")
    ax.plot([0, 1], [n_rows + 0.12, n_rows + 0.12], color="black", linewidth=1.0)
    ax.plot([0, 1], [n_rows + 0.03, n_rows + 0.03], color="black", linewidth=0.7)

    for index, row in df_plot.iterrows():
        y = n_rows - index - 0.5
        is_other = bool(row["is_other"])
        background = "#f2f2f2" if index % 2 == 0 else "#ffffff"
        ax.add_patch(patches.Rectangle((0, y - 0.5), rules_width, 1.0, facecolor=other_color if is_other else background, edgecolor="none"))
        score = row[score_col]
        score_color = other_color if is_other else cmap(norm(score))
        ax.add_patch(patches.Rectangle((score_x_start, y - 0.5), score_width, 1.0, facecolor=score_color, edgecolor="none"))

        if is_other:
            label = next((row[column] for column in condition_cols if row[column] != ""), "")
            ax.text(rules_width / 2, y, label, ha="center", va="center", fontsize=fontsize, color="black", fontweight="bold")
        else:
            for column_index, column in enumerate(condition_cols):
                condition = row[column]
                x_start = x_starts[column_index]
                x_center = x_start + condition_width / 2
                if condition != "":
                    ax.text(x_center, y, condition, ha="center", va="center", fontsize=fontsize, color="black")
                if column_index < max_conditions - 1:
                    left_exists = row[condition_cols[column_index]] != ""
                    right_exists = row[condition_cols[column_index + 1]] != ""
                    if left_exists and right_exists:
                        amp_x = x_start + condition_width + ampersand_width / 2
                        ax.text(amp_x, y, "&", ha="center", va="center", fontsize=fontsize, color="black")

        intensity = abs(score) / vmax
        score_text_color = "black" if is_other or intensity < text_white_threshold else "white"
        ax.text(score_x_start + score_width / 2, y, str(int(round(score))), ha="center", va="center", fontsize=fontsize, color=score_text_color, fontweight="bold" if is_other else "normal")

    ax.add_patch(patches.Rectangle((0, 0), 1, n_rows, fill=False, edgecolor="black", linewidth=0.7))
    ax.plot([score_x_start, score_x_start], [0, n_rows], color="white", linewidth=1.0)

    scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = plt.colorbar(scalar_mappable, cax=cax, orientation="vertical")
    colorbar.set_ticks([-vmax, 0, vmax])
    colorbar.set_ticklabels([f"{-int(vmax)}", "0", f"{int(vmax)}"])
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.tick_params(labelsize=9)
    cax.set_title(bar_label, fontsize=header_fontsize, color="gray", pad=10)
    cax.text(0.5, 1.07, str(classes[1]), transform=cax.transAxes, ha="center", va="bottom", fontsize=header_fontsize, fontweight="bold", color="#008c3a")
    cax.text(0.5, -0.07, str(classes[0]), transform=cax.transAxes, ha="center", va="top", fontsize=header_fontsize, fontweight="bold", color="#e60000")

    if save_path is not None:
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax, df_plot


def save_rulecard_visualization(
    scorecard: Any,
    feature_names: list[str],
    classes: list[Any] | tuple[Any, Any] | np.ndarray,
    output_path: Path,
    *,
    onehot_mapping: dict[str, dict] | None = None,
    boolean_features: list[str] | None = None,
    max_rules: int = 25,
    figsize: tuple[float, float] = (7.0, 7.0),
    row_height: float = 0.30,
    show: bool = False,
) -> pd.DataFrame:
    """Salva il PNG rosso/verde e restituisce le regole leggibili."""
    readable_rules = scorecard_rules_to_df(
        scorecard,
        feature_names,
        onehot_mapping=onehot_mapping,
        boolean_features=boolean_features,
    )
    fig, _, _ = plot_rulecard_table(
        readable_rules,
        classes=classes,
        max_rules=max_rules,
        figsize=figsize,
        row_height=row_height,
        save_path=output_path,
        show=show,
    )
    if not show:
        plt.close(fig)
    return readable_rules
