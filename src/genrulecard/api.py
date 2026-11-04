"""Small public API for the fixed GenRuleCard pipeline."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .core.genrulecard import GenRuleCard as _Pipeline
from .defaults import DEFAULT_CORPUS_SIZE, build_recommended_config


class GenRuleCard:
    """Fit the recommended GenRuleCard configuration on a binary CSV dataset.

    The corpus size is the only methodological parameter exposed publicly.
    Feature types are inferred from pandas dtypes unless explicitly supplied.
    """

    def __init__(
        self,
        dataset: str | Path,
        *,
        target: str,
        positive_class: int | str,
        corpus_size: int = DEFAULT_CORPUS_SIZE,
        output_dir: str | Path = "grc_outputs",
        numeric_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
        boolean_features: list[str] | None = None,
    ) -> None:
        self.dataset = Path(dataset).expanduser().resolve()
        if not self.dataset.is_file():
            raise FileNotFoundError(f"Dataset not found: {self.dataset}")
        if corpus_size < 2:
            raise ValueError("corpus_size must be at least 2.")

        frame = pd.read_csv(self.dataset)
        if target not in frame.columns:
            raise ValueError(f"Target column '{target}' is not present in the dataset.")
        classes = set(frame[target].dropna().unique())
        if len(classes) != 2:
            raise ValueError("GenRuleCard currently supports binary targets only.")
        if positive_class not in classes:
            raise ValueError("positive_class is not present in the target column.")

        supplied = any(value is not None for value in (
            numeric_features, categorical_features, boolean_features
        ))
        if supplied:
            numeric = list(numeric_features or [])
            categorical = list(categorical_features or [])
            boolean = list(boolean_features or [])
        else:
            features = frame.drop(columns=[target])
            boolean = list(features.select_dtypes(include=["bool", "boolean"]).columns)
            numeric = [
                column for column in features.select_dtypes(include="number").columns
                if column not in boolean
            ]
            categorical = [
                column for column in features.columns
                if column not in set(numeric + boolean)
            ]

        selected = numeric + categorical + boolean
        if len(selected) != len(set(selected)):
            raise ValueError("A feature cannot belong to more than one feature group.")
        missing = set(frame.columns).difference(selected + [target])
        if missing:
            raise ValueError(
                "Feature groups do not cover all columns: " + ", ".join(sorted(missing))
            )

        config = build_recommended_config(
            csv_path=self.dataset,
            target=target,
            positive_class=positive_class,
            numeric_features=numeric,
            categorical_features=categorical,
            boolean_features=boolean,
            corpus_size=int(corpus_size),
            output_dir=Path(output_dir),
        )
        self._pipeline = _Pipeline(config)

    @property
    def run_dir(self) -> Path:
        """Directory containing the artifacts of the current run."""
        return self._pipeline.run_dir

    @property
    def scorecard_(self):
        """Selected genetic ScoreCard, available after fit()."""
        if self._pipeline.genetic is None:
            raise RuntimeError("Call fit() before accessing scorecard_.")
        return self._pipeline.genetic.scorecard

    def fit(self) -> "GenRuleCard":
        """Run corpus generation, FMI-VAE training, and genetic search."""
        self._pipeline.fit()
        return self

    def evaluate(self) -> pd.DataFrame:
        """Evaluate the selected models once on the isolated internal test set."""
        return self._pipeline.predict()
