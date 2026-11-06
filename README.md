# Uzbek encoder models: text classification and NER

Code, run-level results and manuscript source for the study
"Encoder models for Uzbek text classification and named entity recognition:
a controlled comparison with tokenization and orthographic-variation analysis"
(XLM-R-base, mBERT, TahrirchiBERT-base/small on SIB-200 `uzn_Latn`, WikiANN-uz and the
Uzbek news corpus `murodbek/uz-text-classification`).

## Layout
- `code/experiments/` the experiment scripts in the order they were run: `01_fixed_budget_benchmark.py` (fixed-budget runs on SIB-200, WikiANN and news), `02_ner_ablation_and_class_analysis.py` (NER apostrophe ablation, per-class and error analyses), `03_news_fairness_v1.py`, `04_news_norm_diag.py` (orthography diagnosis), `05_news_fairness_v2.py` (shared search, 512 tokens, ASCII phase).
- `notebooks/` the same experiments as Kaggle notebooks (T4 GPU); `05b` is the resume variant.
- `code/notebook_sources/` percent-format sources and `build.py`, which composes notebooks 04 and 05 (`python build.py a|b`); `test_*.py` are torch-free logic tests on synthetic data.
- `code/upload_kaggle_dataset.py` helper to publish result folders as private Kaggle datasets.
- `results/` run-level results (`*.jsonl`), summaries, statistical tests, per-run predictions (`preds/`), environment files and `__huggingface_repos__.json` (exact Hugging Face revisions).
- `paper/` LaTeX source (Acta NUUz template), `build_paper.py` (fills tables and numbers from `results/`) and the compiled PDF; `paper/al_fargoniy/` the version formatted for the Al-Farg'oniy avlodlari information letter (Word file, PDF and the scripts that build it).

## Data
Datasets are **not** redistributed here; they are public on Hugging Face (revisions are in the paper's Data availability section and in `results/**/__huggingface_repos__.json`). Check each dataset's licence before reuse.

## Caveats
Small numbers of seeds (3-5), a single pilot seed for configuration search and a fixed news subsample (seed 0); see the Discussion and limitations section of the paper.
