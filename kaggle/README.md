# Kaggle notebook

**Published:** [kaggle.com/code/pannphetra/google-merch-store-the-next-marketing-dollar](https://www.kaggle.com/code/pannphetra/google-merch-store-the-next-marketing-dollar)

[`merch_store_marketing_case_study.ipynb`](merch_store_marketing_case_study.ipynb) is a single, self-contained version of the case study for Kaggle. It downloads the clean session table from BigQuery's public Google Analytics sample. It then reruns three analyses:
- the segments: Google employees (41% of revenue) and the key account, one corporate buyer reported on its own;
- the retargeting model against a funnel rule, a two-line rule, and the lab's features, with its break-even and monthly value (D2);
- the attribution comparison, with the Organic Search gap and the attributed value of a paid click (D1).

Its helper cells are this repository's `src/` modules, copied in by the build script ([`build_notebook.py`](build_notebook.py)), so its results match the full notebooks; [`tests/test_kaggle_notebook.py`](../tests/test_kaggle_notebook.py) fails if the committed copy falls behind the builder. On Kaggle, other library versions can move a few model figures slightly (for example, the top-20% value can differ by about $2 a month). Four parts are summarized rather than re-run, to keep it short: the audit of Google's BigQuery ML lab (recreating BigQuery ML models needs your own Google Cloud project), the retargeting evaluation without hindsight, the range of the Organic Search over-credit, and the test designs (notebooks 03–06).

## Publish it on Kaggle

With the [Kaggle CLI](https://github.com/Kaggle/kaggle-cli), signed in once with `kaggle auth login`, one command uploads the notebook, and Kaggle runs it. [`kernel-metadata.json`](kernel-metadata.json) holds its Kaggle id, title and settings.

```bash
python kaggle/build_notebook.py    # after changing src/, the session SQL or the notebook text
kaggle kernels push -p kaggle
kaggle kernels status pannphetra/google-merch-store-the-next-marketing-dollar
```

Each push saves and runs a new version. The first cell after the helper code queries BigQuery and scans about 0.8 GB, so the notebook runs with internet on, which Kaggle allows only for phone-verified accounts.

Without the CLI:
1. On Kaggle: **Create → New Notebook → File → Import Notebook**, and upload the `.ipynb` file.
2. In the notebook settings, turn **Internet on**.
3. **Run all.** Kaggle notebooks can query BigQuery public datasets with the Python client. If your session asks for credentials, attach a Google Cloud account under **Add-ons → Google Cloud Services**.
4. **Save Version → Save & Run All** so the published version shows its outputs.

Tags: none are set yet. A CLI push sends the `keywords` list in `kernel-metadata.json` as the notebook's tags, so before adding one, check that each tag exists on Kaggle. Candidates: `marketing`, `business`, `bigquery`, `classification`, `data visualization`.
