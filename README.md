# Uzbek encoder models: text classification and NER

Code, run-level results and manuscript source for the study
"Encoder models for Uzbek text classification and named entity recognition:
a controlled comparison with tokenization and orthographic-variation analysis"
(XLM-R-base, mBERT, TahrirchiBERT-base/small on SIB-200 `uzn_Latn`, WikiANN-uz and the
Uzbek news corpus `murodbek/uz-text-classification`).

## Layout
- `notebooks/` Kaggle notebooks (T4 GPU) for the news diagnosis, the shared-search fairness run and its resume variant.
- `code/notebook_sources/` percent-format sources and `build.py`, which composes the notebooks (`python build.py a|b`); `test_*.py` are torch-free logic tests on synthetic data.
- `code/upload_kaggle_dataset.py` helper to publish result folders as private Kaggle datasets.
- `results/` run-level results (`*.jsonl`), summaries, statistical tests, per-run predictions (`preds/`), environment files and `__huggingface_repos__.json` (exact Hugging Face revisions).
- `paper/` LaTeX source (Acta NUUz template), `build_paper.py` (fills tables and numbers from `results/`) and the compiled PDF.

## Data
Datasets are **not** redistributed here; they are public on Hugging Face (revisions are in the paper's Data availability section and in `results/**/__huggingface_repos__.json`). Check each dataset's licence before reuse.

## Caveats
Small numbers of seeds (3-5), a single pilot seed for configuration search and a fixed news subsample (seed 0); see the Discussion and limitations section of the paper.
