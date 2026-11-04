"""End-to-end installation check; intentionally uses a tiny corpus."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer

from genrulecard import GenRuleCard


def test_end_to_end(tmp_path: Path) -> None:
    dataset = load_breast_cancer(as_frame=True).frame
    csv_path = tmp_path / "breast_cancer.csv"
    dataset.to_csv(csv_path, index=False)

    model = GenRuleCard(
        dataset=csv_path,
        target="target",
        positive_class=1,
        corpus_size=20,
        output_dir=tmp_path / "outputs",
    )
    model.fit()
    report = model.evaluate()

    assert not report.empty
    assert {"best_initial_corpus", "best_genetic"} == set(report["candidate"])
    assert model.run_dir.is_dir()
