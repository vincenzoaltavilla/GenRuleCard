from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class DatasetConfig:
    name: str = "titanic"
    csv_path: str | Path | None = None
    target_column: str | None = None
    numeric_features: list[str] | None = None
    categorical_features: list[str] | None = None
    boolean_features: list[str] | None = None
    positive_class: int | str = 1


@dataclass
class SplitConfig:
    train_size: float = 0.70
    validation_size: float = 0.15
    test_size: float = 0.15
    test_random_state: int = 120
    validation_random_state: int = 120

    def __post_init__(self) -> None:
        total = self.train_size + self.validation_size + self.test_size
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Le quote train, validation e test devono sommare a 1.")


@dataclass
class RuleCardConfig:
    n_cards: int = 300
    reuse_latest_corpus: bool = False
    stratified_sampling: bool = True
    balance_sampled_classes: bool = False
    initial_selection_metric: str = "accuracy_f1"
    initial_selection_accuracy_weight: float = 0.5
    s1_choice: str = "pareto"
    s1_f1_weight: float = 0.5
    bootstrap_fraction: float = 1.0
    max_attempts_multiplier: int = 4
    sample_modes: list[str] = field(default_factory=lambda: ["bootstrap", "full_train", "majority_80", "majority_65"])
    learning_rates: list[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50])
    top_k_features: list[str | int] = field(default_factory=lambda: ["sqrt", 3, 5, "all"])
    reuse_features: list[bool] = field(default_factory=lambda: [True])
    feature_fractions: list[float] = field(default_factory=lambda: [1.00, 0.85, 0.65, 0.50])
    tree_max_depth: int = 2
    max_n_iter: int = 20
    patience: int = 5
    use_fast: bool = True
    n_jobs: int = -1
    pdo: int = 100
    odds0: int = 50
    score0: int = 600

    def __post_init__(self) -> None:
        grid_fields = (
            "sample_modes", "learning_rates", "top_k_features",
            "reuse_features", "feature_fractions",
        )
        for name in grid_fields:
            if not getattr(self, name):
                raise ValueError(f"rulecard.{name} non puo essere vuoto.")
        if self.initial_selection_metric not in {"accuracy", "f1", "accuracy_f1"}:
            raise ValueError("rulecard.initial_selection_metric non valido.")
        if self.s1_choice not in {"pareto", "score", "min_rules"}:
            raise ValueError(
                "rulecard.s1_choice deve essere 'pareto', 'score' o 'min_rules'."
            )
        if not 0.0 <= self.initial_selection_accuracy_weight <= 1.0:
            raise ValueError("rulecard.initial_selection_accuracy_weight deve essere tra 0 e 1.")
        if not 0.0 <= self.s1_f1_weight <= 1.0:
            raise ValueError("rulecard.s1_f1_weight deve essere tra 0 e 1.")
    def as_rulecard_params(
        self,
        *,
        learning_rate: float | None = None,
        top_k_features: str | int | float | None = None,
        reuse_features: bool | None = None,
    ) -> dict[str, Any]:
        selected_top_k = self.top_k_features[0] if top_k_features is None else top_k_features
        if selected_top_k == "all":
            selected_top_k = float("inf")
        return {
            "learning_rate": float(self.learning_rates[0] if learning_rate is None else learning_rate),
            "max_n_iter": self.max_n_iter,
            "patience": self.patience,
            "use_fast": self.use_fast,
            "reuse_features": bool(self.reuse_features[0] if reuse_features is None else reuse_features),
            "top_k_features": selected_top_k,
            "n_jobs": self.n_jobs,
        }


@dataclass
class VAEConfig:
    representation: str = "row"
    # "standard" usa esattamente loss_weights; "auto" modifica soltanto il
    # peso globale della pattern loss dopo avere osservato il corpus.
    vae_profile: str = "standard"
    auto_pattern_weight_mode: str = "discrete"
    reuse_latest_vae: bool = False
    # Opzionale: inizializza il training da un checkpoint compatibile. Se
    # ``fine_tune_original_corpus_size`` e' valorizzato, l'holdout 80/20 viene
    # ricostruito sulle sole ScoreCard originarie e le nuove entrano nel train.
    initial_checkpoint: Path | str | None = None
    fine_tune_original_corpus_size: int | None = None
    # Modello consegnato alle diagnostiche e al GA. ``earlystop`` usa il
    # checkpoint scelto sullo split interno 80/20 e salta il refit su tutto T.
    deployment_checkpoint: str = "final_refit"
    reconstruction_diagnostic_scope: str = "all"
    notebook_checkpoint_root: Path | str | None = None
    latent_dim: int = 16
    hidden_dim: int = 256
    hidden_dims: tuple[int, ...] | None = None
    n_layers: int = 3
    n_heads: int = 4
    dropout_p: float = 0.15
    batch_size: int = 48
    epochs: int = 350
    learning_rate: float = 8e-4
    weight_decay: float = 3e-4
    patience: int = 40
    min_delta: float = 1e-4
    lr_patience: int = 18
    min_lr: float = 1e-5
    early_stopping_monitor: str = "total"
    decoded_monitor_interval: int = 5
    decoded_monitor_cards: int = 12
    decoded_monitor_min_epochs: int = 40
    decoded_reconstruction_weight: float = 0.25
    decoded_predictive_weight: float = 2.0
    beta_max: float = 0.002
    beta_warmup_fraction: float = 0.35
    metric_target: str = "structural"
    metric_alpha: float = 0.7
    structural_distance_mode: str = "matrix"
    partition_probe_size: int = 5000
    distance_n_jobs: int = 1
    decode_count_penalty: float = 0.0
    tree_monitor_metric_weight: float = 0.20
    tree_monitor_local_metric_weight: float = 0.10
    tree_local_metric_temperature: float = 0.20
    signature_temperature: float = 0.35
    metric_pearson_weight: float = 0.50
    metric_rank_weight: float = 0.50
    metric_rank_temperature: float = 0.10
    metric_rank_min_gap: float = 0.02
    refit_max_epochs: int = 500
    refit_min_epochs: int = 200
    refit_patience: int = 80
    refit_min_delta: float = 1e-4
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "pattern": 3.0, "padding": 2.0, "thr1": 2.0, "thr2": 1.5,
        "weight": 2.0, "prior": 0.5, "metric": 0.55, "local_metric": 0.0, "count": 0.0,
    })

    def __post_init__(self) -> None:
        if self.distance_n_jobs < 1:
            raise ValueError("vae.distance_n_jobs deve essere almeno 1.")
        if (
            self.fine_tune_original_corpus_size is not None
            and self.fine_tune_original_corpus_size < 2
        ):
            raise ValueError(
                "vae.fine_tune_original_corpus_size deve essere almeno 2."
            )
        if self.representation not in {"row", "tree"}:
            raise ValueError("vae.representation deve essere 'row' oppure 'tree'.")
        if self.deployment_checkpoint not in {"final_refit", "earlystop"}:
            raise ValueError("vae.deployment_checkpoint deve essere 'final_refit' oppure 'earlystop'.")
        if self.reconstruction_diagnostic_scope not in {"all", "holdout"}:
            raise ValueError("vae.reconstruction_diagnostic_scope deve essere 'all' oppure 'holdout'.")
        if self.vae_profile not in {"standard", "auto", "fmi_f1_sig"}:
            raise ValueError("vae.vae_profile deve essere 'standard', 'auto' oppure 'fmi_f1_sig'.")
        allowed_pattern_modes = {"discrete", "smooth_soft", "smooth_aggressive"}
        if self.auto_pattern_weight_mode not in allowed_pattern_modes:
            raise ValueError(
                f"auto_pattern_weight_mode deve essere uno tra {sorted(allowed_pattern_modes)}."
            )
        allowed_monitors = {"total", "metric", "decoded_composite"}
        if self.early_stopping_monitor not in allowed_monitors:
            raise ValueError(
                f"early_stopping_monitor deve essere uno tra {sorted(allowed_monitors)}."
            )
        if self.decoded_monitor_interval < 1 or self.decoded_monitor_cards < 1:
            raise ValueError("Gli intervalli e il numero di anchor del monitor decoded devono essere >= 1.")
        if not 1 <= self.decoded_monitor_min_epochs <= self.epochs:
            raise ValueError("decoded_monitor_min_epochs deve essere tra 1 ed epochs.")
        if self.decoded_reconstruction_weight < 0 or self.decoded_predictive_weight < 0:
            raise ValueError("I pesi del monitor decoded devono essere >= 0.")
        allowed_targets = {"structural", "predictive", "mixed"}
        if self.metric_target not in allowed_targets:
            raise ValueError(f"metric_target deve essere uno tra {sorted(allowed_targets)}.")
        if not 0.0 <= self.metric_alpha <= 1.0:
            raise ValueError("metric_alpha deve essere compreso tra 0 e 1.")
        allowed_distance_modes = {"matrix", "semantic", "semantic_matching", "partition_fmi"}
        if self.structural_distance_mode not in allowed_distance_modes:
            raise ValueError(
                f"structural_distance_mode deve essere uno tra {sorted(allowed_distance_modes)}."
            )
        if self.partition_probe_size < 1:
            raise ValueError("partition_probe_size deve essere almeno 1.")
        if self.decode_count_penalty < 0.0:
            raise ValueError("decode_count_penalty deve essere >= 0.")
        if self.n_layers < 1:
            raise ValueError("vae.n_layers deve essere almeno 1.")
        if self.hidden_dims is not None and (not self.hidden_dims or any(value < 1 for value in self.hidden_dims)):
            raise ValueError("vae.hidden_dims deve contenere dimensioni positive.")
        if self.n_heads < 1:
            raise ValueError("vae.n_heads deve essere almeno 1.")
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("vae.hidden_dim deve essere divisibile per vae.n_heads.")
        if self.tree_monitor_metric_weight < 0.0 or self.tree_monitor_local_metric_weight < 0.0:
            raise ValueError("I pesi del monitor TreeVAE devono essere >= 0.")
        if self.tree_local_metric_temperature <= 0.0:
            raise ValueError("vae.tree_local_metric_temperature deve essere > 0.")
        if self.signature_temperature <= 0.0 or self.metric_rank_temperature <= 0.0:
            raise ValueError("Le temperature del profilo VAE devono essere > 0.")
        if self.refit_min_epochs < 1 or self.refit_max_epochs < self.refit_min_epochs:
            raise ValueError("vae.refit_max_epochs deve essere >= refit_min_epochs >= 1.")
        if self.refit_patience < 1 or self.refit_min_delta < 0.0:
            raise ValueError("Parametri di early stopping del refit non validi.")


@dataclass
class GeneticConfig:
    # Seed opzionale del solo GA; split, corpus e VAE continuano a usare
    # ExperimentConfig.seed.
    random_seed: int | None = None
    population_size: int = 160
    steps: int = 50
    elite_size: int = 48
    mutation_std: float = 0.18
    mutation_decay: float = 0.985
    crossover_low: float = 0.15
    crossover_high: float = 0.85
    patience: int | None = None
    min_delta: float = 0.0
    fitness_metric: str = "accuracy"
    accuracy_f1_accuracy_weight: float = 0.5
    f1_complexity_f1_weight: float = 0.5
    f1_complexity_initial_f1_weight: float | None = None
    f1_complexity_rule_tolerance: float = 0.25
    # "distance": vicinanza al numero iniziale; "min_rules": premia
    # direttamente meno regole; "target_reduction": premio saturato al target;
    # "target_constraint": il peso complexity determina la riduzione richiesta;
    # "absolute_min_rules": usa n_rules / massimo rappresentabile, senza S1.
    f1_complexity_rule_objective: str = "distance"
    f1_complexity_target_fraction: float = 0.50
    initial_anchor_fraction: float = 0.0
    confidence_weight: float = 0.0
    complexity_weight: float = 0.0
    redundancy_weight: float = 0.0
    validation_folds: int = 1
    validation_folds_dynamic: bool = False
    validation_stability_weight: float = 0.0
    deterministic: bool = False
    save_reproducibility_artifacts: bool = False
    fitness_round_decimals: int = 12
    checkpoint_interval: int = 0
    # Valutazioni indipendenti degli individui della popolazione.
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.random_seed is not None and self.random_seed < 0:
            raise ValueError("genetic.random_seed deve essere None oppure >= 0.")
        if self.n_jobs < 1:
            raise ValueError("genetic.n_jobs deve essere almeno 1.")
        if self.elite_size <= 1 or self.elite_size > self.population_size:
            raise ValueError("elite_size deve essere tra 2 e population_size.")
        if self.patience is not None and self.patience < 1:
            raise ValueError("genetic.patience deve essere None oppure almeno 1.")
        if self.min_delta < 0.0:
            raise ValueError("genetic.min_delta deve essere >= 0.")
        allowed_fitness = {"accuracy", "f1", "accuracy_f1", "f1_complexity"}
        if self.fitness_metric not in allowed_fitness:
            raise ValueError(f"fitness_metric deve essere uno tra {sorted(allowed_fitness)}.")
        if not 0.0 <= self.accuracy_f1_accuracy_weight <= 1.0:
            raise ValueError("genetic.accuracy_f1_accuracy_weight deve essere compreso tra 0 e 1.")
        if not 0.0 <= self.f1_complexity_f1_weight <= 1.0:
            raise ValueError("genetic.f1_complexity_f1_weight deve essere compreso tra 0 e 1.")
        if self.f1_complexity_initial_f1_weight is not None and not 0.0 <= self.f1_complexity_initial_f1_weight <= 1.0:
            raise ValueError("genetic.f1_complexity_initial_f1_weight deve essere None oppure compreso tra 0 e 1.")
        if self.f1_complexity_rule_tolerance < 0.0:
            raise ValueError("genetic.f1_complexity_rule_tolerance deve essere >= 0.")
        if self.f1_complexity_rule_objective not in {
            "distance", "min_rules", "target_reduction", "target_constraint",
            "absolute_min_rules",
        }:
            raise ValueError(
                "genetic.f1_complexity_rule_objective deve essere 'distance', "
                "'min_rules', 'target_reduction', 'target_constraint' oppure "
                "'absolute_min_rules'."
            )
        if not 0.0 < self.f1_complexity_target_fraction < 1.0:
            raise ValueError(
                "genetic.f1_complexity_target_fraction deve essere in (0, 1)."
            )
        if not 0.0 <= self.initial_anchor_fraction <= 1.0:
            raise ValueError("genetic.initial_anchor_fraction deve essere compreso tra 0 e 1.")
        if self.validation_folds < 1:
            raise ValueError("genetic.validation_folds deve essere almeno 1.")
        if self.validation_stability_weight < 0.0:
            raise ValueError("genetic.validation_stability_weight deve essere >= 0.")
        if self.fitness_round_decimals < 0 or self.fitness_round_decimals > 15:
            raise ValueError("genetic.fitness_round_decimals deve essere tra 0 e 15.")
        if self.checkpoint_interval < 0:
            raise ValueError("genetic.checkpoint_interval deve essere >= 0.")
        for name in ("confidence_weight", "complexity_weight", "redundancy_weight"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"genetic.{name} deve essere >= 0.")


@dataclass
class VisualizationConfig:
    """Parametri della grafica RuleCard rosso/verde."""
    scorecard_max_rules: int = 25
    scorecard_figsize: tuple[float, float] = (7.0, 7.0)
    scorecard_row_height: float = 0.30
    show_scorecard_windows: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 2026
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    rulecard: RuleCardConfig = field(default_factory=RuleCardConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    genetic: GeneticConfig = field(default_factory=GeneticConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output_root: Path | str = Path("experiments")
    rulecard_source_dir: Path | str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_root"] = str(self.output_root)
        data["rulecard_source_dir"] = (
            str(self.rulecard_source_dir) if self.rulecard_source_dir else None
        )
        data["vae"]["notebook_checkpoint_root"] = (
            str(self.vae.notebook_checkpoint_root)
            if self.vae.notebook_checkpoint_root else None
        )
        return data
