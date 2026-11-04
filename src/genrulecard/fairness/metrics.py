from __future__ import annotations

import numpy as np
import pandas as pd


FAIRNESS_FEATURES = {
    "adult": ("sex", "race"),
    "compas": ("sex", "race"),
    "titanic": ("sex",),
    "telco": ("gender",),
    "taiwan_credit": ("sex",),
    "german_credit": ("sex",),
}


def sensitive_values(dataset: str, raw: pd.DataFrame, feature: str) -> np.ndarray:
    dataset = str(dataset)
    if dataset == "taiwan_credit" and feature == "sex":
        return raw["x2"].map({1: "male", 2: "female"}).fillna("unknown").astype(str).to_numpy()
    if dataset == "german_credit" and feature == "sex":
        return raw["personal_status"].astype(str).map(
            lambda value: "female" if value.lower().startswith("female") else "male"
        ).to_numpy()
    if dataset == "compas" and feature == "race":
        african = pd.to_numeric(raw["race_African_American"], errors="coerce").fillna(0).to_numpy() > 0.5
        caucasian = pd.to_numeric(raw["race_Caucasian"], errors="coerce").fillna(0).to_numpy() > 0.5
        return np.where(african, "African-American", np.where(caucasian, "Caucasian", "Other"))
    return raw[feature].astype(str).to_numpy()


def group_fairness(y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive = np.asarray(sensitive).astype(str)
    selection_rates, tprs, fprs = [], [], []
    for group in np.unique(sensitive):
        mask = sensitive == group
        actual_positive = y_true[mask] == 1
        predicted_positive = y_pred[mask] == 1
        actual_negative = ~actual_positive
        selection_rates.append(float(predicted_positive.mean()))
        if actual_positive.any():
            tprs.append(float(predicted_positive[actual_positive].mean()))
        if actual_negative.any():
            fprs.append(float(predicted_positive[actual_negative].mean()))

    def difference(values: list[float]) -> float:
        return float(max(values) - min(values)) if len(values) >= 2 else np.nan

    dpd = difference(selection_rates)
    eop = difference(tprs)
    fpr_difference = difference(fprs)
    eodd = max(eop, fpr_difference) if np.isfinite(eop) and np.isfinite(fpr_difference) else np.nan
    return {
        "demographic_parity_difference": dpd,
        "equal_opportunity_difference": eop,
        "equalized_odds_difference": float(eodd),
    }


def fairness_penalty(metrics: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    dp_violation = max(0.0, metrics["demographic_parity_difference"] - reference["median_dpd"])
    eodd_violation = max(0.0, metrics["equalized_odds_difference"] - reference["median_eodd"])
    penalty = 0.35 * dp_violation + 0.65 * eodd_violation
    return {
        "fairness_dpd_violation": float(dp_violation),
        "fairness_eodd_violation": float(eodd_violation),
        "fairness_penalty": float(penalty),
    }
