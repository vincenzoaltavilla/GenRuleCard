"""Apply the fixed GenRuleCard pipeline to a local company dataset."""
from genrulecard import GenRuleCard


model = GenRuleCard(
    dataset="company_dataset.csv",
    target="target",
    positive_class=1,
    corpus_size=10_000,
    output_dir="outputs/company_dataset",
)

model.fit()
print(model.evaluate())
print(f"Artifacts saved in: {model.run_dir}")
