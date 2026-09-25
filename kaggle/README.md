# Kaggle notebook

[`merch_store_marketing_case_study.ipynb`](merch_store_marketing_case_study.ipynb) is a single, self-contained version of the case study for Kaggle. It downloads the clean session table from BigQuery's public Google Analytics sample. It then reruns three analyses:
- the employee-traffic finding;
- the retargeting model and its break-even (D2);
- the attribution comparison (D1).

Its helper cells are this repository's `src/` modules, copied in by the build script, so its results match the full notebooks. The audit of Google's BigQuery ML lab is summarized at the end rather than re-run, because recreating BigQuery ML models needs your own Google Cloud project.

## Publish it on Kaggle

1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and upload the `.ipynb` file.
2. In the notebook settings, turn **Internet on**. The first cell after the helper code queries BigQuery and scans about 0.8 GB.
3. **Run all.** Kaggle notebooks can query BigQuery public datasets with the Python client. If your session asks for credentials, attach a Google Cloud account under **Add-ons → Google Cloud Services**.
4. **Save Version → Save & Run All** so the published version shows its outputs.

Suggested tags: `marketing`, `google analytics`, `bigquery`, `attribution`, `classification`.
