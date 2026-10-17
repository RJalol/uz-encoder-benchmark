# %% [markdown]
# # News fairness v2: 512 token + kengaytirilgan qidiruv (yakuniy adolat tekshiruvi)
# **v1 dan ma'lum bo'lgan cheklovlar, shu notebook ularni yopadi:**
# 1. `max_len=256` da hujjatlarning 54–82% qirqilgan, saqlangan matn ulushi 0.59 (mBERT) … 0.78 (Tahrirchi) → noteng. Bu yerda **max_len=512**
#    (tokenizerlarning umumiy chegarasi; baribir qisman qirqish qoladi, diagnostika `truncation_diagnostics_512.csv` da).
# 2. v1 da XLM-R va mBERT tanlangan konfiguratsiyasi qidiruv chegarasida va `best_epoch` ≈ oxirgi epoch edi (ehtimol hali o'qitilmagan).
#    Bu yerda LR ∈ {3e-5, 5e-5, 1e-4}, 6 epoch (+ eng yaxshi checkpoint validatsiya bo'yicha); `best_epoch == epochs` bo'lsa hisobotda belgilanadi.
# 3. Pilot 1 seed edi → yakuniy seedlar `FINAL_SEEDS` (default 3; vaqt bo'lsa 5 ga oshiring — resume-safe).
# **Protokol:** barcha 4 model uchun bir xil qidiruv maydoni, tanlash faqat validatsiya Macro-F1; test faqat yakuniy hisobot uchun.
# **Apostrof normalizatsiyasi:** diag notebook topilmasi (Jinoyat sinfi tipografik belgi `\`` ga tayanadi) tufayli, tanlangan konfiguratsiya
# ikkinchi bosqichda ASCII-normalizatsiyalangan matnda ham o'tkaziladi (qayta sozlashsiz). Birinchi bosqich (orig) to'liq tugagach boshlanadi.
# **Batch:** 16 × grad.accum 2 = effektiv 32 (v1 bilan bir xil), xotira uchun.
# **Vaqt:** T4 da orig bosqichi ~9–10 soat, ASCII bosqichi yana ~5–6 soat (jami ~15 soat = 2 sessiya) (SMOKE=False). `TIME_BUDGET_H` dan oshsa to'xtaydi; keyingi sessiyada oldingi chiqishni Input qilib ulang (resume).
# Kaggle: Internet On, GPU T4 x1. Avval `SMOKE = True`.

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

OUT = "/kaggle/working/news_fairness_v2"
os.makedirs(f"{OUT}/preds", exist_ok=True)
ENV = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
       "datasets": datasets.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(ENV)
if not torch.cuda.is_available(): print("!!! GPU YO'Q: Accelerator'ni GPU T4 ga qo'ying !!!")
json.dump(ENV, open(f"{OUT}/environment_news_fairness_v2.json", "w"), indent=2)

# %%
# ---------------- CONFIG ----------------
SMOKE = False                     # to'liq run (davom ettirish). Sinov uchun True.
RESUME_MIN = 23                   # davom ettirishda kamida shuncha oldingi yozuv topilishi shart (0 = noldan boshlash). Dataset ulanmasa, 10 soatlik pilot behuda qayta ishlamasin.
PILOT_SEEDS = [13]
FINAL_SEEDS = [13, 21, 42]        # 5 ga oshirish: [13, 21, 42, 87, 100] (resume-safe)
TIME_BUDGET_H = 11.0              # Kaggle sessiya limiti 12 soat; shundan oshsa xavfsiz to'xtaydi
MODELS = {
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "tahrirchi-base": "tahrirchi/tahrirchi-bert-base",
    "tahrirchi-small": "tahrirchi/tahrirchi-bert-small",
}
BASE = dict(bs=16, ga=2, max_len=512)
CONFIGS = {                       # BARCHA modellar uchun bir xil; tartib = "soddaroq" birinchi (teng bo'lsa shu tanlanadi)
    "e_lr3": dict(lr=3e-5, epochs=6, cond="orig"),
    "e_lr5": dict(lr=5e-5, epochs=6, cond="orig"),
    "e_lr1e4": dict(lr=1e-4, epochs=6, cond="orig"),
}
N_TRAIN, N_VAL, N_TEST = 10000, 2000, 5000
if SMOKE:
    FINAL_SEEDS = FINAL_SEEDS[:2]
    N_TRAIN, N_VAL, N_TEST = 500, 200, 200
    for c in CONFIGS.values(): c["epochs"] = 1
