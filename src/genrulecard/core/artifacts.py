from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class ProcessedDataset:
    name: str
    X_train: np.ndarray
    X_validation: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    feature_min: np.ndarray
    feature_max: np.ndarray
    feature_range: np.ndarray
    model_classes: np.ndarray
    raw_split: dict[str, pd.DataFrame]
    metadata: dict


@dataclass
class CorpusBundle:
    corpus_df: pd.DataFrame
    cards_df: pd.DataFrame
    tree_rows_df: pd.DataFrame
    pattern_vocab: pd.DataFrame
    card_ids: list[int]
    scorecard_tensor: np.ndarray
    pat_target: np.ndarray
    thr_target: np.ndarray
    weight_target: np.ndarray
    pad_target: np.ndarray
    has_child: np.ndarray
    real_mask: np.ndarray
    thr1_mask: np.ndarray
    thr2_mask: np.ndarray
    weight_mask: np.ndarray
    prior_log_odds: np.ndarray
    prior_target: np.ndarray
    n_sc: int
    t_max: int
    n_rows: int
    n_pat: int
    row_dim: int
    weight_mean: float
    weight_std: float
    prior_mean: float
    prior_std: float
    prior_min: float
    prior_max: float
    best_initial_id: int
    best_initial_validation_accuracy: float


@dataclass
class VAEArtifacts:
    model: object
    embeddings: np.ndarray
    best_epoch: int
    earlystop_history: pd.DataFrame
    final_history: pd.DataFrame
    representation_report: pd.DataFrame
    latent_pca: pd.DataFrame
    structural_distance: np.ndarray
    latent_distance: np.ndarray
    predictive_distance: np.ndarray


@dataclass
class GeneticResult:
    best_z: np.ndarray
    validation_accuracy: float
    rows: pd.DataFrame
    leaves: pd.DataFrame
    prior_log_odds: float
    scorecard: object
    history: pd.DataFrame
    info: dict


@dataclass
class RunPaths:
    root: Path
    rulecard: Path
    vae: Path
    genetic: Path
    figures: Path
    results: Path
