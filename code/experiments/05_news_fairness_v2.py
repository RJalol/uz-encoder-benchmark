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
# ---------------- TRUNCATION DIAGNOSTIKASI (max_len=512) ----------------
rows = []; sample = TE.text.tolist()[:1000]
for mk, mn in MODELS.items():
    tok = load_tok(mn)
    lens = np.array([len(x) for x in tok(sample, add_special_tokens=True, truncation=False)["input_ids"]])
    kept = np.minimum(BASE["max_len"], lens) / lens
    rows.append(dict(model=mk, max_len=BASE["max_len"], tokens_per_doc=lens.mean(), docs_truncated=(lens > BASE["max_len"]).mean(), text_kept_fraction=kept.mean()))
TRUNC = pd.DataFrame(rows); TRUNC.to_csv(f"{OUT}/truncation_diagnostics_512.csv", index=False)
print(TRUNC.round(3).to_string(index=False))

# %%
# ---------------- MAIN: pilot -> tanlash -> yakuniy (resume-safe, vaqt nazorati bilan) ----------------
RES = f"{OUT}/news_v2_results{'_smoke' if SMOKE else ''}.jsonl"
RECS = []
for f in ([RES] + ([] if SMOKE else [p for p in glob.glob("/kaggle/input/**/news_v2_results.jsonl", recursive=True)])):   # oldingi sessiyadan davom ettirish
    if os.path.exists(f):
        for l in open(f):
            r = json.loads(l)
            if not any(x["cfg"] == r["cfg"] and x["model"] == r["model"] and x["seed"] == r["seed"] for x in RECS): RECS.append(r)
if RECS and not os.path.exists(RES): open(RES, "w").write("".join(json.dumps(r, default=float) + "\n" for r in RECS))
print("mavjud yozuvlar (resume):", len(RECS))
if not SMOKE and len(RECS) < RESUME_MIN:
    raise RuntimeError(f"Oldingi natijalar topilmadi ({len(RECS)} < {RESUME_MIN}). 'uz-news-v2-part1' datasetini Add Input orqali ulang va qayta ishga tushiring. "
                       "Noldan boshlamoqchi bo'lsangiz, RESUME_MIN = 0 qiling.")
T0 = time.time(); STOP = False

def ensure(cfg_name, mkey, seed):
    global STOP
    if any(r["cfg"] == cfg_name and r["model"] == mkey and r["seed"] == seed for r in RECS): return True
    if (time.time() - T0) / 3600 > TIME_BUDGET_H:
        STOP = True; print(f"VAQT LIMITI ({TIME_BUDGET_H} soat): to'xtadi. Keyingi sessiyada shu chiqishni Input qilib ulab qayta ishga tushiring."); return False
    print(f"=== {cfg_name} | {mkey} | seed={seed} | {(time.time() - T0) / 3600:.2f} soat", flush=True)
    try:
        rec = run_cfg(cfg_name, mkey, seed)
    except Exception as e:
        open(f"{OUT}/errors.log", "a").write(f"{cfg_name} {mkey} {seed}: {repr(e)}\n"); print("XATO:", repr(e)[:300])
        shutil.rmtree(f"{OUT}/tmp_run", ignore_errors=True); gc.collect(); torch.cuda.empty_cache(); return False
    print({k: (round(rec[k], 4) if isinstance(rec[k], float) else rec[k]) for k in ("val_macro_f1", "test_macro_f1", "n_never_pred", "best_epoch")})
    RECS.append(rec); open(RES, "a").write(json.dumps(rec, default=float) + "\n"); return True

PILOT_CFGS = [c for c in CONFIGS if not c.endswith("__ascii")]   # cell qayta ishga tushsa ham ASCII konfiguratsiyalar pilotga kirmaydi
for cfg_name in PILOT_CFGS:
    for mkey in MODELS:
        for seed in PILOT_SEEDS:
            if STOP: break
            ensure(cfg_name, mkey, seed)
PIL = pd.DataFrame([r for r in RECS if r["seed"] in PILOT_SEEDS and r["cfg"] in PILOT_CFGS])   # resume: "__ascii" yozuvlari pilotga kirmasin
have_all = {m: set(PIL[PIL.model == m].cfg) == set(PILOT_CFGS) for m in MODELS} if len(PIL) else {}
assert have_all and all(have_all.values()), f"pilot to'liq emas (davom ettiring): {have_all}"
SEL, AVG = select_configs(PIL, PILOT_CFGS, PILOT_SEEDS)
PV = AVG.pivot(index="model", columns="cfg", values="val_macro_f1"); PV.to_csv(f"{OUT}/pilot_val_macro_f1_v2.csv")
json.dump(SEL, open(f"{OUT}/selected_configs_v2.json", "w"), indent=2)
print("\nPilot validatsiya Macro-F1:\n", PV.round(4).to_string(), "\nTanlangan:", SEL)

for seed in FINAL_SEEDS:
    for mkey in MODELS:
        if STOP: break
        ensure(SEL[mkey], mkey, seed)
# 2-bosqich: xuddi shu tanlangan konfiguratsiya, apostrof-ASCII normalizatsiyalangan matnda (qayta sozlashsiz).
# Maqsad: news'dagi tipografik signal (masalan, Jinoyat sinfidagi "`") bo'lmaganda modellarni adolatli solishtirish (diag notebook topilmasi).
ASC = {}
for mkey in MODELS:
    ASC[mkey] = SEL[mkey] + "__ascii"; CONFIGS[ASC[mkey]] = {**CONFIGS[SEL[mkey]], "cond": "norm_ascii"}
for seed in FINAL_SEEDS:
    for mkey in MODELS:
        if STOP: break
        ensure(ASC[mkey], mkey, seed)
print(f"\nJami vaqt: {(time.time() - T0) / 3600:.2f} soat | STOP={STOP}")

# %%
# ---------------- AGGREGATION ----------------
R = pd.DataFrame(RECS); assert len(R) > 0, "natijalar yo'q: errors.log ni tekshiring"
def pick(names_by_model):
    parts = [R[(R.model == m) & (R.cfg == c) & (R.seed.isin(FINAL_SEEDS))] for m, c in names_by_model.items()]
    return pd.concat(parts) if parts else R.iloc[:0]
fin = pick(SEL)                      # orig matn
fin_a = pick(ASC) if "ASC" in globals() else R.iloc[:0]   # ASCII-normalizatsiyalangan matn
ms = sorted(fin.model.unique())
def summarize(df, label):
    rows = []
    for m, g in df.groupby("model"):
        rows.append(dict(setting=label, model=m, cfg=g.cfg.iloc[0], lr=g.lr.iloc[0], n_seeds=len(g), test_macro_f1=g.test_macro_f1.mean(),
                         std=g.test_macro_f1.std(ddof=1) if len(g) > 1 else np.nan, test_accuracy=g.test_accuracy.mean(),
                         seeds_with_collapse=int((g.n_never_pred > 0).sum()), val_macro_f1=g.val_macro_f1.mean(),
                         best_epoch_mean=g.best_epoch.mean(), best_epoch_at_last=int((g.best_epoch >= g.epochs - 0.01).sum()),
                         lr_at_grid_edge=bool(g.lr.iloc[0] in (min(c["lr"] for c in CONFIGS.values() if c["cond"] == "orig"), max(c["lr"] for c in CONFIGS.values() if c["cond"] == "orig")))))
    return rows
S = pd.DataFrame(summarize(fin, "orig") + (summarize(fin_a, "norm_ascii") if len(fin_a) else []))
S.to_csv(f"{OUT}/news_v2_summary.csv", index=False)
print(S.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
print("\nESLATMA: best_epoch_at_last>0 yoki lr_at_grid_edge=True bo'lsa, model hali chegarada — 'to'liq adolatli' deb da'vo qilmang.")

def model_tests(df, label):
    pairs = []; mm = sorted(df.model.unique())
    for i in range(len(mm)):
        for j in range(i + 1, len(mm)):
            a, b = df[df.model == mm[i]].test_macro_f1, df[df.model == mm[j]].test_macro_f1
            if len(a) > 1 and len(b) > 1:
                t, p = stats.ttest_ind(a, b, equal_var=False); pairs.append(dict(setting=label, A=mm[i], B=mm[j], mean_diff=a.mean() - b.mean(), p=p))
    if pairs:
        P = pd.DataFrame(pairs); P["p_holm"] = holm(P.p.values); return P
    return pd.DataFrame()
PT = pd.concat([model_tests(fin, "orig")] + ([model_tests(fin_a, "norm_ascii")] if len(fin_a) else []), ignore_index=True)
if len(PT):
    PT.to_csv(f"{OUT}/news_v2_tests.csv", index=False)
    print("\nModellar orasida (Welch + Holm, har setting ichida; n kichik, ehtiyot bo'ling):\n", PT.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
# orig -> ASCII: har model uchun seed-juftlik farqi
if len(fin_a):
    rows = []
    for m in ms:
        a = fin[fin.model == m].set_index("seed").test_macro_f1; b = fin_a[fin_a.model == m].set_index("seed").test_macro_f1
        c = a.index.intersection(b.index)
        if len(c) > 1:
            d = (b[c] - a[c]); t, p = stats.ttest_rel(b[c], a[c]); rows.append(dict(model=m, n_pairs=len(c), mean_diff_ascii_minus_orig=d.mean(), p=p))
    if rows:
        N = pd.DataFrame(rows); N["p_holm"] = holm(N.p.values); N.to_csv(f"{OUT}/news_v2_orig_vs_ascii.csv", index=False)
        print("\nASCII normalizatsiyasining ta'siri (juft t-test + Holm):\n", N.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

# v1 (max_len=256, tuned) bilan solishtirish: truncation ta'siri (config'lar farq qilgani uchun bu "512 + kengaytirilgan qidiruv" ning umumiy ta'siri)
if V1_RES and not SMOKE:
    V1 = pd.read_json(io.StringIO(open(V1_RES[0]).read()), lines=True)
    sel1 = json.load(open(glob.glob("/kaggle/input/**/selected_configs.json", recursive=True)[0])) if glob.glob("/kaggle/input/**/selected_configs.json", recursive=True) else {}
    cmp_ = []
    for m in ms:
        if m in sel1:
            a = V1[(V1.model == m) & (V1.cfg == sel1[m])].test_macro_f1; b = fin[fin.model == m].test_macro_f1
            cmp_.append(dict(model=m, v1_cfg=sel1[m], v1_256_f1=a.mean(), v2_512_f1=b.mean(), diff=b.mean() - a.mean()))
    if cmp_:
        C = pd.DataFrame(cmp_); C.to_csv(f"{OUT}/v1_vs_v2.csv", index=False); print("\nv1 (256) vs v2 (512):\n", C.round(4).to_string(index=False))
# sinf bo'yicha F1 (orig va ASCII); Jinoyat sinfiga alohida e'tibor (tipografik signal)
tab = pd.DataFrame({"class": NAMES, "n_train": TRAIN_CNT})
for lab, df in [("orig", fin), ("ascii", fin_a)]:
    pc = df[df.per_class_f1.notna()]
    for m, g in pc.groupby("model"): tab[f"{m}_{lab}_f1"] = np.array(g.per_class_f1.tolist()).mean(0)
tab.to_csv(f"{OUT}/news_v2_per_class_f1.csv", index=False)
rare = tab.n_train <= 110
print("\nKam sinflar o'rtacha F1:", {c: round(tab.loc[rare, c].mean(), 3) for c in tab.columns if c.endswith("_f1")})
print(tab.round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(7, 4)); s = fin.groupby("model").test_macro_f1.agg(["mean", "std"]).reindex(ms)
ax.bar(range(len(ms)), s["mean"], yerr=s["std"], capsize=3); ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_ylabel("news test Macro-F1 (512)")
plt.tight_layout(); plt.savefig(f"{OUT}/news_v2_macro_f1.png", dpi=130); plt.close()
print("\nTAYYOR:", OUT)
