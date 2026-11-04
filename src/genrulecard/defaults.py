"""Internal configuration for the released GenRuleCard method.

Only the corpus size is intentionally variable. The remaining values reproduce
the configuration selected in the experimental study.
"""
from __future__ import annotations

import os
from pathlib import Path

from .core.config import (
    DatasetConfig,
    ExperimentConfig,
    GeneticConfig,
    RuleCardConfig,
    SplitConfig,
    VAEConfig,
    VisualizationConfig,
)


DEFAULT_CORPUS_SIZE = 10_000
GLOBAL_ALPHA = 0.7
SEED = 32


def build_recommended_config(
    *,
    csv_path: Path,
    target: str,
    positive_class: int | str,
    numeric_features: list[str],
    categorical_features: list[str],
    boolean_features: list[str],
    corpus_size: int,
    output_dir: Path,
) -> ExperimentConfig:
    """Create the fixed configuration selected for GenRuleCard."""
    workers = max(1, os.cpu_count() or 1)
    return ExperimentConfig(
        seed=SEED,
        dataset=DatasetConfig(
            name="csv",
            csv_path=csv_path,
            target_column=target,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            boolean_features=boolean_features,
            positive_class=positive_class,
        ),
        split=SplitConfig(
            train_size=0.70,
            validation_size=0.15,
            test_size=0.15,
            test_random_state=SEED,
            validation_random_state=SEED,
        ),
        rulecard=RuleCardConfig(
            n_cards=corpus_size,
            reuse_latest_corpus=False,
            stratified_sampling=True,
            balance_sampled_classes=True,
            initial_selection_metric="f1",
            initial_selection_accuracy_weight=0.35,
            s1_choice="pareto",
            s1_f1_weight=0.5,
            bootstrap_fraction=1.0,
            max_attempts_multiplier=10,
            sample_modes=["bootstrap", "majority_80", "majority_65"],
            learning_rates=[0.20, 0.35, 0.50],
            top_k_features=["sqrt", "25pct", "50pct"],
            reuse_features=[False, True],
            feature_fractions=[0.50, 0.75, 1.00],
            tree_max_depth=2,
            max_n_iter=15,
            patience=5,
            use_fast=True,
            n_jobs=workers,
            pdo=100,
            odds0=50,
            score0=600,
        ),
        vae=VAEConfig(
            representation="tree",
            vae_profile="fmi_f1_sig",
            reuse_latest_vae=False,
            deployment_checkpoint="earlystop",
            reconstruction_diagnostic_scope="holdout",
            latent_dim=96,
            hidden_dim=384,
            hidden_dims=(384, 384, 192),
            n_layers=3,
            n_heads=6,
            dropout_p=0.10,
            batch_size=32,
            epochs=150,
            learning_rate=3e-4,
            weight_decay=1e-5,
            patience=15,
            min_delta=0.005,
            lr_patience=18,
            min_lr=1e-5,
            early_stopping_monitor="total",
            beta_max=0.005,
            beta_warmup_fraction=0.0,
            metric_target="structural",
            metric_alpha=0.5,
            structural_distance_mode="partition_fmi",
            partition_probe_size=2000,
            distance_n_jobs=workers,
            tree_monitor_metric_weight=0.5,
            tree_monitor_local_metric_weight=0.0,
            loss_weights={
                "type": 1.0,
                "feature": 1.0,
                "threshold": 0.8,
                "weight": 0.8,
                "prior": 0.2,
                "count_leaves": 0.0,
                "signature_count": 0.0,
                "f1": 0.0,
                "metric": 3.0,
                "local_metric": 0.0,
            },
        ),
        genetic=GeneticConfig(
            random_seed=SEED,
            population_size=100,
            steps=100,
            elite_size=15,
            mutation_std=0.35,
            mutation_decay=0.997,
            crossover_low=0.05,
            crossover_high=0.95,
            patience=25,
            min_delta=0.002,
            fitness_metric="f1_complexity",
            accuracy_f1_accuracy_weight=0.35,
            f1_complexity_f1_weight=GLOBAL_ALPHA,
            f1_complexity_rule_objective="absolute_min_rules",
            deterministic=False,
            fitness_round_decimals=12,
            checkpoint_interval=0,
            n_jobs=workers,
        ),
        visualization=VisualizationConfig(show_scorecard_windows=False),
        output_root=output_dir,
    )
