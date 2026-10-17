# %% [markdown]
# # News: apostrof normalizatsiyasi diagnostikasi (c4_norm nima uchun yomonlashdi?)
# **Kuzatuv:** fairness v1 pilotida `c4_norm` (apostrof-o'xshash belgilar → ASCII `'`) barcha modellarda `c0` dan yomon chiqdi
# (Tahrirchi-base val Macro-F1 0.794 → 0.727), holbuki NER'da bu normalizatsiya deyarli hech narsani o'zgartirmagan.
# **Gipotezalar:**
# - H1: pipeline farqi: `c0` asosiy benchmarkdan olingan, `c4` yangi kodda o'qitilgan (normalizatsiya emas, boshqa narsa farq qiladi).
# - H2: shortcut: apostrof varianti manba/sinf bilan bog'liq (har bir nashr o'z tipografiyasi) va normalizatsiya shu signalni o'chiradi.
# - H3: NFC yoki tokenizatsiya ta'siri (masalan, ASCII `'` ni tokenizator boshqacha bo'ladi).
# **Dizayn:** (1) o'qitishsiz diagnostika: apostrof sensusi, sinf bo'yicha taqsimot, TF-IDF+LR (orig vs norm) va faqat-apostrof klassifikatori;
# (2) tez fine-tuning kontrol: `d_orig` (yangi kodda qayta) va `d_ascii` (NFC + ASCII), 2 seed, hamma model (`d_nfc` ixtiyoriy: RUN_NFC).
# Qaror qoidasi oldindan: agar `d_orig` ≈ `c0` (asosiy benchmark) va `d_ascii` < `d_orig` seed-juftlari bo'yicha izchil → normalizatsiya haqiqatan zararli.
# Agar `d_orig` ≠ `c0` → H1 (pipeline farqi), c0/c4 solishtirishlari ishonchsiz.
# Kaggle: Internet On, GPU T4 x1. Avval `SMOKE = True`. Taxminiy vaqt (SMOKE=False, d_nfc'siz): ~1.5–2 soat.

# %%
import os, sys, json, glob, time, gc, shutil, math, platform, unicodedata, re, io, dataclasses
from collections import Counter
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support
import torch, transformers, datasets
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, set_seed)

OUT = "/kaggle/working/news_diag"
os.makedirs(f"{OUT}/preds", exist_ok=True)
ENV = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
       "datasets": datasets.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(ENV)
if not torch.cuda.is_available(): print("!!! GPU YO'Q: Accelerator'ni GPU T4 ga qo'ying, aks holda juda sekin bo'ladi !!!")
json.dump(ENV, open(f"{OUT}/environment_news_diag.json", "w"), indent=2)

# %%
# ---------------- CONFIG ----------------
SMOKE = True                      # To'liq run uchun False
SEEDS = [13, 21]
MODELS = {
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "tahrirchi-base": "tahrirchi/tahrirchi-bert-base",
    "tahrirchi-small": "tahrirchi/tahrirchi-bert-small",
}
BASE = dict(bs=32, ga=1, max_len=256)           # asosiy benchmark bilan bir xil
CONFIGS = {                                      # faqat matn sharti farq qiladi; lr/epoch = asosiy benchmark (2e-5, 3)
    "d_orig":  dict(lr=2e-5, epochs=3, cond="orig"),
    "d_ascii": dict(lr=2e-5, epochs=3, cond="norm_ascii"),
}
RUN_NFC = False                                  # smoke NFC hech narsani o'zgartirmasligini ko'rsatdi; True bo'lsa d_nfc ham o'qitiladi
if RUN_NFC: CONFIGS["d_nfc"] = dict(lr=2e-5, epochs=3, cond="nfc")
N_TRAIN, N_VAL, N_TEST = 10000, 2000, 5000
if SMOKE:
    N_TRAIN, N_VAL, N_TEST = 500, 200, 200
    for c in CONFIGS.values(): c["epochs"] = 1
