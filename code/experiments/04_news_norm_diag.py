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

# %%
# ---------------- PURE FUNCTIONS (torch'siz) ----------------
APOS_CHARS = ["’", "‘", "ʼ", "ʻ", "´", "`", "′", "ʹ"]
CENSUS_CHARS = ["'"] + APOS_CHARS

def norm_nfc(s): return unicodedata.normalize("NFC", s)
def norm_apos(s, target="'"):
    s = unicodedata.normalize("NFC", s)
    for c in APOS_CHARS:
        s = s.replace(c, target)
    return s
TEXT_FN = {"orig": lambda t: t, "nfc": norm_nfc, "norm_ascii": norm_apos}

def holm(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj

def select_configs(df, order, pilot_seeds):
    """Har model uchun pilot seedlar bo'yicha o'rtacha VALIDATSIYA Macro-F1 eng yuqori cfg. Teng bo'lsa `order` dagi oldingisi. Test ishlatilmaydi."""
    d = df[df.seed.isin(pilot_seeds)]
    avg = d.groupby(["model", "cfg"]).val_macro_f1.mean().reset_index()
    sel = {}
    for m, g in avg.groupby("model"):
        g = g.assign(o=g.cfg.map({c: i for i, c in enumerate(order)})).sort_values(["val_macro_f1", "o"], ascending=[False, True])
        sel[m] = g.iloc[0].cfg
    return sel, avg

def class_stats(y, pred, K):
    f1 = precision_recall_fscore_support(y, pred, labels=list(range(K)), zero_division=0)[2]
    return [float(x) for x in f1], int((np.bincount(pred, minlength=K) == 0).sum())

def apos_counts(text):
    """Har apostrof-o'xshash belgi uchun hujjatdagi soni (ASCII ' va APOS_CHARS)."""
    return [text.count(c) for c in CENSUS_CHARS]

assert norm_apos("o‘zbek ma’no gʻalaba o`zbek") == "o'zbek ma'no g'alaba o'zbek"
assert apos_counts("o'zbek gʻalaba ma’no")[:1] == [1] and sum(apos_counts("o'zbek gʻalaba ma’no")) == 3
_df = pd.DataFrame([dict(model="m", cfg="c0", seed=1, val_macro_f1=.5), dict(model="m", cfg="c1", seed=1, val_macro_f1=.5),
                    dict(model="m", cfg="c2", seed=1, val_macro_f1=.7), dict(model="m", cfg="c2", seed=2, val_macro_f1=.1)])
assert select_configs(_df, ["c0", "c1", "c2"], [1])[0]["m"] == "c2"
assert select_configs(_df.iloc[:2], ["c0", "c1"], [1])[0]["m"] == "c0"
assert abs(holm([0.01, 0.04, 0.03])[0] - 0.03) < 1e-12
print("pure-function self-tests OK")

# %%
# ---------------- DATA (asosiy benchmark bilan bir xil subsample: shuffle seed=0) ----------------
D = load_dataset("murodbek/uz-text-classification")
NAMES = [str(x) for x in D["train"].features["label"].names]; K = len(NAMES)
def prep(split, n):
    s = D[split].shuffle(seed=0).select(range(min(n, len(D[split]))))
    return pd.DataFrame({"text": list(s["text"]), "label": [int(x) for x in s["label"]]})
TR, VA, TE = prep("train", N_TRAIN), prep("validation", N_VAL), prep("test", N_TEST)
TRAIN_CNT = np.bincount(TR.label, minlength=K)
print({"train": len(TR), "val": len(VA), "test": len(TE)}, "| sinflar:", K)
Y_TEST = TE.label.values
np.save(f"{OUT}/test_labels{'_smoke' if SMOKE else ''}.npy", Y_TEST)
# oldingi natijalar (ixtiyoriy Input): asosiy benchmark va/yoki news_fairness v1
MAIN_RES = glob.glob("/kaggle/input/**/results.jsonl", recursive=True)
V1_RES = glob.glob("/kaggle/input/**/news_tuned_results.jsonl", recursive=True)
print("Input: asosiy benchmark results.jsonl:", MAIN_RES[:1], "| v1 news_tuned_results.jsonl:", V1_RES[:1])

# %%
# ---------------- TRAINING (umumiy) ----------------
_VALID_TA = {f.name for f in dataclasses.fields(TrainingArguments)}

def common_args(lr, epochs, bs, ga, seed, outdir, n_train):
    total_steps = math.ceil(n_train / (bs * ga)) * epochs
    kw = dict(output_dir=outdir, per_device_train_batch_size=bs, gradient_accumulation_steps=ga, per_device_eval_batch_size=32, learning_rate=lr,
              num_train_epochs=epochs, weight_decay=0.01, warmup_steps=int(0.1 * total_steps),
              eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
              metric_for_best_model="macro_f1", greater_is_better=True, fp16=torch.cuda.is_available(),
              seed=seed, data_seed=seed, report_to="none", logging_steps=100, dataloader_num_workers=2, disable_tqdm=True)
    dropped = [k for k in kw if k not in _VALID_TA]
    if dropped: print("OGOHLANTIRISH: TrainingArguments tashlab yuborilgan parametrlar:", dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in _VALID_TA})

def cls_metrics(p):
    pred = p.predictions.argmax(-1); y = p.label_ids
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, pred, average="macro", zero_division=0)}

def run_cfg(cfg_name, mkey, seed):
    cfg = {**BASE, **CONFIGS[cfg_name]}; mname = MODELS[mkey]; outdir = f"{OUT}/tmp_run"
    set_seed(seed); tok = load_tok(mname)
    f = TEXT_FN[cfg["cond"]]
    def enc(df):
        e = dict(tok([f(t) for t in df.text], truncation=True, max_length=cfg["max_len"])); e["labels"] = df.label.tolist()
        return Dataset.from_dict(e)
    tr, va, te = enc(TR), enc(VA), enc(TE)
    model = AutoModelForSequenceClassification.from_pretrained(mname, num_labels=K)
    n_params = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model=model, args=common_args(cfg["lr"], cfg["epochs"], cfg["bs"], cfg["ga"], seed, outdir, len(tr)), train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok), processing_class=tok, compute_metrics=cls_metrics)
    t0 = time.time(); trainer.train(); train_s = time.time() - t0
    bv = trainer.evaluate(va); out = trainer.predict(te); pred = out.predictions.argmax(-1)
    pcf1, never = class_stats(Y_TEST, pred, K)
    best_epoch = None
    ck = getattr(trainer.state, "best_model_checkpoint", None)
    if ck and re.search(r"checkpoint-(\d+)", ck):
        best_epoch = round(int(re.search(r"checkpoint-(\d+)", ck).group(1)) / math.ceil(len(tr) / (cfg["bs"] * cfg["ga"])), 2)
    np.save(f"{OUT}/preds/{cfg_name}_{mkey}_{seed}{'_smoke' if SMOKE else ''}.npy", pred)
    rec = dict(cfg=cfg_name, model=mkey, seed=seed, lr=cfg["lr"], epochs=cfg["epochs"], cond=cfg["cond"], max_len=cfg["max_len"], params=n_params,
               train_time_s=round(train_s, 1), peak_mem_mb=int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
               best_epoch=best_epoch, val_macro_f1=float(bv["eval_macro_f1"]), val_accuracy=float(bv["eval_accuracy"]),
               test_macro_f1=float(out.metrics["test_macro_f1"]), test_accuracy=float(out.metrics["test_accuracy"]),
               test_weighted_f1=float(out.metrics["test_weighted_f1"]), test_macro_precision=float(out.metrics["test_macro_precision"]),
               test_macro_recall=float(out.metrics["test_macro_recall"]), n_never_pred=never, per_class_f1=pcf1, source="trained")
    del trainer, model; shutil.rmtree(outdir, ignore_errors=True); gc.collect(); torch.cuda.empty_cache()
    return rec

def load_tok(name):
    tok = AutoTokenizer.from_pretrained(name)
    try:
        tok(["a", "b"], is_split_into_words=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        tok(["a", "b"], is_split_into_words=True)
        print(f"[{name}] add_prefix_space=True bilan yuklandi")
    return tok

# %%
# ---------------- 1) O'QITISHSIZ DIAGNOSTIKA: apostrof sensusi ----------------
ALL = pd.concat([TR.assign(split="train"), VA.assign(split="val"), TE.assign(split="test")], ignore_index=True)
cnt = np.array([apos_counts(t) for t in ALL.text])
CEN = pd.DataFrame({"char": CENSUS_CHARS, "codepoint": [f"U+{ord(c):04X}" for c in CENSUS_CHARS],
                    "name": [unicodedata.name(c, "?") for c in CENSUS_CHARS],
                    "occurrences": cnt.sum(0), "docs_with": (cnt > 0).sum(0), "frac_docs": (cnt > 0).mean(0)})
CEN.to_csv(f"{OUT}/apostrophe_census_news.csv", index=False)
print("Apostrof sensusi (barcha splitlar):\n", CEN.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
changed_nfc = np.mean([norm_nfc(t) != t for t in ALL.text]); changed_ascii = np.mean([norm_apos(t) != t for t in ALL.text])
print(f"\nhujjatlarning NFC o'zgartirgan ulushi: {changed_nfc:.4f} | ASCII-normalizatsiya o'zgartirgan ulushi: {changed_ascii:.4f}")

# sinf bo'yicha: har belgi variantini ishlatadigan hujjatlar ulushi (shortcut belgisi)
by_class = pd.DataFrame({"class": NAMES, "n_docs": np.bincount(ALL.label, minlength=K)})
for j, c in enumerate(CENSUS_CHARS):
    by_class["frac_" + ("ASCII" if c == chr(39) else f"U+{ord(c):04X}")] = [float((cnt[ALL.label.values == k, j] > 0).mean()) for k in range(K)]
by_class.to_csv(f"{OUT}/apostrophe_by_class.csv", index=False)
print("\nSinf bo'yicha (hujjat ulushi):\n", by_class.round(3).to_string(index=False))
# split o'rtasida taqsimot farqi
print("\nSplit bo'yicha (hujjat ulushi):\n", pd.DataFrame({s: (cnt[ALL.split.values == s] > 0).mean(0) for s in ["train", "val", "test"]}, index=CENSUS_CHARS).round(4).to_string())

# %%
# ---------------- 2) SHORTCUT TESTI (CPU, tez): TF-IDF+LR orig vs norm; faqat-apostrof klassifikator ----------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
def tfidf_lr(fn, analyzer="word"):
    # analyzer="word": token_pattern \w+ tinish belgilarini tashlaydi -> apostrof variantlarini KO'RMAYDI (faqat nazorat uchun).
    # analyzer="char": belgi n-gramlar apostrof variantlarini saqlaydi -> shortcut'ni haqiqatan sinaydi.
    kw = dict(analyzer="char", ngram_range=(1, 4), min_df=2, max_features=300000) if analyzer == "char" else dict(token_pattern=r"(?u)\b\w+\b", min_df=2, max_features=200000)
    v = TfidfVectorizer(preprocessor=lambda t: fn(t).lower(), sublinear_tf=True, **kw)
    Xtr = v.fit_transform(TR.text); lr_ = LogisticRegression(max_iter=300, C=10).fit(Xtr, TR.label)
    return {"val_macro_f1": f1_score(VA.label, lr_.predict(v.transform(VA.text)), average="macro"),
            "test_macro_f1": f1_score(TE.label, lr_.predict(v.transform(TE.text)), average="macro")}
SC = {"tfidf_word_orig": tfidf_lr(TEXT_FN["orig"]), "tfidf_word_norm_ascii": tfidf_lr(TEXT_FN["norm_ascii"]),
      "tfidf_char_orig": tfidf_lr(TEXT_FN["orig"], "char"), "tfidf_char_norm_ascii": tfidf_lr(TEXT_FN["norm_ascii"], "char")}
# faqat apostrof belgilari: [log1p soni] bo'yicha klassifikator (token mazmuni YO'Q)
Atr, Ate = np.log1p(np.array([apos_counts(t) for t in TR.text])), np.log1p(np.array([apos_counts(t) for t in TE.text]))
maj = np.full(len(TE), np.bincount(TR.label).argmax())
SC["apostrophe_only_LR"] = {"val_macro_f1": np.nan, "test_macro_f1": f1_score(TE.label, LogisticRegression(max_iter=500).fit(Atr, TR.label).predict(Ate), average="macro")}
SC["majority_class"] = {"val_macro_f1": np.nan, "test_macro_f1": f1_score(TE.label, maj, average="macro")}
SC = pd.DataFrame(SC).T; SC.to_csv(f"{OUT}/shortcut_check.csv")
print("Shortcut testi (test Macro-F1):\n", SC.round(4).to_string())
print("Talqin: apostrophe_only_LR >> majority_class bo'lsa, apostrof varianti sinf haqida ma'lumot beradi (shortcut). "
      "tfidf_char_orig >> tfidf_char_norm_ascii bo'lsa, normalizatsiya foydali signalni o'chiradi (word-variant apostrofni ko'rmaydi, faqat nazorat).")

# %%
# ---------------- 3) TOKENIZATSIYA TA'SIRI (orig vs norm_ascii) ----------------
rows = []; sample = TE.text.tolist()[:500]
for mk, mn in MODELS.items():
    tok = load_tok(mn)
    for cond in ["orig", "nfc", "norm_ascii"]:
        enc_ = tok([TEXT_FN[cond](t) for t in sample], add_special_tokens=False)["input_ids"]
        rows.append(dict(model=mk, cond=cond, mean_tokens=np.mean([len(x) for x in enc_]),
                         unk_rate=np.mean([sum(i == tok.unk_token_id for i in ids) / max(len(ids), 1) for ids in enc_])))
TK = pd.DataFrame(rows); TK.to_csv(f"{OUT}/tokenization_effect.csv", index=False)
print(TK.to_string(index=False, float_format=lambda x: f"{x:.5g}"))

# %%
# ---------------- 4) FINE-TUNING KONTROL (resume-safe) ----------------
RES = f"{OUT}/diag_results{'_smoke' if SMOKE else ''}.jsonl"
RECS = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
print("Natijalar fayli:", RES, "| mavjud yozuvlar:", len(RECS))
T0 = time.time()
for seed in SEEDS:
    for mkey in MODELS:
        for cfg_name in CONFIGS:
            if any(r["cfg"] == cfg_name and r["model"] == mkey and r["seed"] == seed for r in RECS): continue
            print(f"=== {cfg_name} | {mkey} | seed={seed} | {(time.time() - T0) / 3600:.2f} soat", flush=True)
            try:
                rec = run_cfg(cfg_name, mkey, seed)
            except Exception as e:
                open(f"{OUT}/errors.log", "a").write(f"{cfg_name} {mkey} {seed}: {repr(e)}\n"); print("XATO:", repr(e)[:300])
                shutil.rmtree(f"{OUT}/tmp_run", ignore_errors=True); gc.collect(); torch.cuda.empty_cache(); continue
            print({k: (round(rec[k], 4) if isinstance(rec[k], float) else rec[k]) for k in ("val_macro_f1", "test_macro_f1", "n_never_pred", "best_epoch")})
            RECS.append(rec); open(RES, "a").write(json.dumps(rec, default=float) + "\n")

# %%
# ---------------- 5) TAHLIL ----------------
R = pd.DataFrame(RECS); assert len(R) > 0, "natijalar yo'q: errors.log ni tekshiring"
piv = lambda col: R.pivot_table(index=["model", "seed"], columns="cfg", values=col)
T, V = piv("test_macro_f1"), piv("val_macro_f1")
for nm, df in [("test_macro_f1", T), ("val_macro_f1", V)]:
    d = df.copy()
    d["ascii-orig"] = d.d_ascii - d.d_orig
    if "d_nfc" in d: d["nfc-orig"] = d.d_nfc - d.d_orig
    print(f"\n{nm} (model x seed):\n", d.round(4).to_string())
    d.to_csv(f"{OUT}/diag_{nm}_by_seed.csv")
NEV = R.pivot_table(index=["model", "seed"], columns="cfg", values="n_never_pred")
print("\nHech bashorat qilinmagan sinflar soni:\n", NEV.to_string())
M = T.groupby("model").mean(); M["ascii-orig"] = M.d_ascii - M.d_orig
if "d_nfc" in M: M["nfc-orig"] = M.d_nfc - M.d_orig
print("\nModel bo'yicha o'rtacha test Macro-F1 va farqlar:\n", M.round(4).to_string()); M.to_csv(f"{OUT}/diag_summary.csv")

# d_orig ni asosiy benchmark (c0) bilan solishtirish (bir xil seed -> teng bo'lishi kerak)
if MAIN_RES and not SMOKE:
    MAIN = pd.read_json(io.StringIO(open(MAIN_RES[0]).read()), lines=True); MAIN = MAIN[MAIN.dataset == "uz_news"]
    cmp_ = []
    for _, r in R[R.cfg == "d_orig"].iterrows():
        m = MAIN[(MAIN.model == r.model) & (MAIN.seed == r.seed)]
        if len(m) == 1: cmp_.append(dict(model=r.model, seed=r.seed, d_orig_test_f1=r.test_macro_f1, main_c0_test_f1=float(m.test_macro_f1.iloc[0]),
                                         diff=r.test_macro_f1 - float(m.test_macro_f1.iloc[0])))
    CMP = pd.DataFrame(cmp_); CMP.to_csv(f"{OUT}/d_orig_vs_main_benchmark.csv", index=False)
    print("\nd_orig (yangi kod) vs asosiy benchmark c0 (bir xil seed):\n", CMP.round(4).to_string(index=False))
    print("Agar |diff| ~ 0 (<=0.005) -> pipeline bir xil (H1 rad); katta farq -> H1 (pipeline/determinizm farqi).")
else:
    print("\nAsosiy benchmark results.jsonl Input sifatida ulanmagan (yoki SMOKE): d_orig vs c0 solishtirish o'tkazib yuborildi.")

fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(M)); w = 0.35
for k, c in enumerate([c for c in ["d_orig", "d_nfc", "d_ascii"] if c in M]): ax.bar(x + k * w, M[c], w, label=c)
ax.set_xticks(x + w); ax.set_xticklabels(M.index); ax.set_ylabel("news test Macro-F1 (seed o'rtacha)"); ax.legend(); plt.tight_layout()
plt.savefig(f"{OUT}/diag_cond_macro_f1.png", dpi=130); plt.close()
print("\nTAYYOR:", OUT)
