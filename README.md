# GenRuleCard

GenRuleCard generates an interpretable scorecard for binary classification by combining a
corpus of RuleCard models, an FMI-regularized variational autoencoder, and genetic search.

The public implementation intentionally exposes only the input dataset and the corpus size.
All methodological settings use the configuration selected in the experimental study:

- 10,000 scorecards by default;
- FMI-regularized VAE;
- global fitness weight `alpha = 0.7`;
- fixed 70/15/15 train, validation, and test split;
- fixed VAE and genetic-search hyperparameters.

## Installation

GenRuleCard currently targets Python 3.10.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

The pipeline also requires the `RuleCard` Python package at commit
`ecd890e12cbd0e6e6197a7d7463e17c4e8015187`. Until RuleCard provides an installable package,
copy its `RuleCard/` module into the active environment's `site-packages` directory.

## Usage

```python
from genrulecard import GenRuleCard

model = GenRuleCard(
    dataset="company_dataset.csv",
    target="target",
    positive_class=1,
    corpus_size=10_000,
    output_dir="outputs/company_dataset",
)

model.fit()
report = model.evaluate()

print(report)
print(model.run_dir)
```

Numeric, categorical, and boolean features are inferred from the CSV dtypes. They may be
specified explicitly only when the automatic inference is unsuitable. The target must be binary.

`corpus_size` is the only exposed methodological parameter. A smaller value is useful for an
installation check, while 10,000 is the recommended configuration.

## Output

Each run stores the generated corpus, VAE checkpoints and diagnostics, GA history, selected
scorecard, validation metrics, and the final one-time test evaluation under `output_dir`.

## Smoke test

The end-to-end test uses a corpus of 20 scorecards only to verify that the installation works:

```bash
pytest -q tests/test_smoke.py
```

It is not a scientific benchmark and does not change the released default configuration.
