from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from ..core.config import RuleCardConfig


def import_rulecard(source_dir: Path | str | None = None):
    """Import lazy: consente di aprire e ispezionare il progetto senza RuleCard."""
    if source_dir is not None:
        source_dir = Path(source_dir)
        if source_dir.exists() and str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
    try:
        from RuleTree import RuleTreeRegressor
        from RuleCard.RuleCardGAM import RuleCardGAM
        from RuleCard.ScoreCard import ScoreCard
        return RuleTreeRegressor, RuleCardGAM, ScoreCard
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "RuleCard / RuleTree non trovati. Installa le dipendenze con "
            "'python -m pip install -r requirements.txt' nell'interprete del "
            "progetto, oppure imposta rulecard_source_dir in main.py."
        ) from error


def extract_leaves(rcgam, X_ref: np.ndarray, feature_names: list[str], scorecard_id: int, scorecard_cls) -> pd.DataFrame:
    rows: list[dict] = []
    lr = float(rcgam.learning_rate)
    canonical_tree_idx = 0
    for real_idx, estimator, gamma_map in rcgam.estimators_:
        tree_rules = scorecard_cls._get_rules(estimator.root, real_idx, gamma_map, X_ref)
        if not tree_rules:
            continue
        for local_rule_idx, (conditions, _, support, leaf_log_odds) in enumerate(tree_rules):
            conditions_key = tuple((int(f), float(t), str(op)) for f, t, op in conditions)
            rows.append({
                "scorecard_id": int(scorecard_id),
                "tree_idx": int(canonical_tree_idx),
                "local_rule_idx": int(local_rule_idx),
                "conditions_key": conditions_key,
                "rule": " AND ".join(
                    f"{feature_names[int(f)]} {op} {float(t):.8g}" for f, t, op in conditions_key
                ),
                "w": float(leaf_log_odds) * lr,
                "support": int(support),
            })
        canonical_tree_idx += 1
    return pd.DataFrame(rows)


def valid_rulecard(leaves_df: pd.DataFrame) -> bool:
    if leaves_df.empty:
        return False
    if not leaves_df["conditions_key"].map(lambda path: 1 <= len(path) <= 2).all():
        return False
    leaves_per_tree = leaves_df.groupby("tree_idx").size()
    return bool(leaves_per_tree.between(2, 4).all())


def scorecard_from_leaves(
    leaves_df: pd.DataFrame,
    base_log_odds: float,
    classes: np.ndarray,
    config: RuleCardConfig,
    scorecard_cls,
):
    required = {"conditions_key", "w"}
    missing = required.difference(leaves_df.columns)
    if missing:
        raise ValueError(f"Colonne mancanti per ScoreCard: {sorted(missing)}")
    rules = []
    for _, leaf in leaves_df.iterrows():
        support = int(leaf["support"]) if "support" in leaves_df.columns and pd.notna(leaf["support"]) else 0
        rule = [(int(f), float(t), str(op)) for f, t, op in leaf["conditions_key"]]
        rules.append((rule, None, support, float(leaf["w"])))
    scorecard = scorecard_cls(
        rules=rules, lr=1.0, base_log_odds=float(base_log_odds), classes=np.asarray(classes),
        PDO=config.pdo, odds0=config.odds0, score0=config.score0, aggregate=True,
    )
    if len(scorecard.scorecard) == 0:
        raise ValueError("Scorecard vuota: tutte le regole hanno punti arrotondati pari a zero.")
    return scorecard


def scorecard_outputs(scorecard, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(scorecard.predict_proba(X), dtype=np.float64)
    predictions = np.asarray(scorecard.predict(X), dtype=np.int64)
    return probabilities, predictions


def fit_rulecard_scorecard(
    X_boot: np.ndarray,
    y_boot: np.ndarray,
    seed: int,
    config: RuleCardConfig,
    source_dir: Path | str | None,
):
    rule_tree_regressor, rulecard_gam, scorecard_cls = import_rulecard(source_dir)
    rcgam = rulecard_gam(
        **config.as_rulecard_params(),
        base_estimator=rule_tree_regressor(max_depth=config.tree_max_depth),
        random_state=seed,
        verbose=False,
    )
    rcgam.fit(X_boot, y_boot)
    scorecard = scorecard_cls.from_rulecard(
        rcgam, X_boot, PDO=config.pdo, odds0=config.odds0, score0=config.score0,
    )
    return rcgam, scorecard, scorecard_cls
