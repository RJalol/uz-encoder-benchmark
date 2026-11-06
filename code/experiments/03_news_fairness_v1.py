# %% [markdown]
# # News klassifikatsiyasi: adolatli taqqoslash (har model uchun validatsiya bo'yicha sozlash)
# **Muammo:** asosiy benchmarkda news uchun hamma modelga bitta byudjet berilgan edi (LR 2e-5, 3 epoch). Natijada mBERT (5 seedning 4 tasida)
# va XLM-R (2 tasida) 1-2 sinfni umuman bashorat qilmadi, Tahrirchi esa hech birida. Bu farq modelning imkoniyatimi yoki sozlanmagan
# o'qitish byudjetimi, shuni aniqlaymiz.
#
# **Protokol (barcha 4 model uchun bir xil qidiruv maydoni):**
# 1. *Pilot* (seed 13): 5 konfiguratsiya (LR, epoch, apostrof normalizatsiyasi). Tanlash faqat **validatsiya Macro-F1** bo'yicha.
# 2. Har model uchun eng yaxshi konfiguratsiya tanlanadi (test ishlatilmaydi).
# 3. *Yakuniy*: tanlangan konfiguratsiya 5 seedda; test natijalari + sinf kollapsi (hech qachon bashorat qilinmagan sinflar) hisoboti.
#
# **Kirish (tavsiya etiladi):** asosiy benchmark natijalari (`results.jsonl`, `preds/*.npy`) Kaggle **Input** sifatida ulangan bo'lsa,
# bazaviy konfiguratsiya (c0) qayta o'qitilmaydi (taxminan 2 soat tejaladi). Aks holda c0 ham o'qitiladi.
#
# Kaggle sozlamalari: Internet = On, GPU (P100 yoki T4 x1). Avval `SMOKE = True` bilan tekshiring.

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

OUT = "/kaggle/working/news_fairness"
os.makedirs(f"{OUT}/preds", exist_ok=True)
ENV = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
       "datasets": datasets.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(ENV)
json.dump(ENV, open(f"{OUT}/environment_news_fairness.json", "w"), indent=2)

# %%
# ---------------- CONFIG ----------------
SMOKE = True                      # True: tezkor tekshiruv (kichik subset, 1 epoch). To'liq run uchun False.
SEEDS = [13, 21, 42, 87, 100]     # asosiy benchmark bilan bir xil
PILOT_SEEDS = [13]                # sozlash uchun seedlar (ko'proq = ishonchliroq, lekin vaqt ko'proq)
MODELS = {
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "tahrirchi-base": "tahrirchi/tahrirchi-bert-base",
    "tahrirchi-small": "tahrirchi/tahrirchi-bert-small",
}
BASE = dict(bs=32, max_len=256)   # asosiy benchmarkdagi kabi (max_len=256 token; modellar orasida teng emas, pastga qarang)
CONFIGS = {                       # BARCHA modellar uchun bir xil qidiruv maydoni
    "c0_base":   dict(lr=2e-5, epochs=3, cond="orig"),         # asosiy benchmarkdagi konfiguratsiya
    "c1_lr5":    dict(lr=5e-5, epochs=3, cond="orig"),
    "c2_ep6":    dict(lr=3e-5, epochs=6, cond="orig"),
    "c3_lr5ep6": dict(lr=5e-5, epochs=6, cond="orig"),
    "c4_norm":   dict(lr=2e-5, epochs=3, cond="norm_ascii"),   # apostrof-o'xshash belgilar -> ASCII '
}
N_TRAIN, N_VAL, N_TEST = 10000, 2000, 5000    # asosiy benchmark bilan bir xil subsample (shuffle seed=0)
if SMOKE:
    SEEDS = SEEDS[:2]
    N_TRAIN, N_VAL, N_TEST = 500, 200, 200
    for c in CONFIGS.values():
        c["epochs"] = 1
REUSE_BASELINE = not SMOKE        # c0 natijalarini asosiy benchmarkdan olish (Input ulangan bo'lsa)

# %%
# ---------------- PURE FUNCTIONS (torch'siz; alohida sinalgan) ----------------
APOS_CHARS = ["’", "‘", "ʼ", "ʻ", "´", "`", "′", "ʹ"]

def norm_apos(s, target="'"):
    s = unicodedata.normalize("NFC", s)
    for c in APOS_CHARS:
        s = s.replace(c, target)
    return s

def holm(p):
    p = np.asarray(p, dtype=float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj

def select_configs(df, order, pilot_seeds):
    """df: ustunlar model, cfg, seed, val_macro_f1. Har model uchun pilot seedlar bo'yicha o'rtacha val Macro-F1 eng yuqori cfg.
    Teng bo'lsa `order` dagi oldingi (oddiyroq) cfg tanlanadi. Test ishlatilmaydi."""
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

# o'z-o'zini tekshirish
assert norm_apos("o‘zbek ma’no gʻalaba o`zbek") == "o'zbek ma'no g'alaba o'zbek"
_df = pd.DataFrame([dict(model="m", cfg="c0", seed=1, val_macro_f1=.5), dict(model="m", cfg="c1", seed=1, val_macro_f1=.5),
                    dict(model="m", cfg="c2", seed=1, val_macro_f1=.7), dict(model="m", cfg="c2", seed=2, val_macro_f1=.1)])
assert select_configs(_df, ["c0", "c1", "c2"], [1])[0]["m"] == "c2"          # pilot seed 1 bo'yicha c2 yaxshi
assert select_configs(_df.iloc[:2], ["c0", "c1"], [1])[0]["m"] == "c0"       # teng -> oddiyroq (oldingi)
assert abs(holm([0.01, 0.04, 0.03])[0] - 0.03) < 1e-12
print("pure-function self-tests OK")

# %%
# ---------------- DATA (asosiy benchmark bilan bir xil subsample) ----------------
D = load_dataset("murodbek/uz-text-classification")
NAMES = [str(x) for x in D["train"].features["label"].names]; K = len(NAMES)
def prep(split, n):
    s = D[split].shuffle(seed=0).select(range(min(n, len(D[split]))))
    return pd.DataFrame({"text": list(s["text"]), "label": [int(x) for x in s["label"]]})
TR, VA, TE = prep("train", N_TRAIN), prep("validation", N_VAL), prep("test", N_TEST)
TRAIN_CNT = np.bincount(TR.label, minlength=K)
print({"train": len(TR), "val": len(VA), "test": len(TE)}, "| sinflar:", K)
print(pd.DataFrame({"class": NAMES, "n_train": TRAIN_CNT, "n_val": np.bincount(VA.label, minlength=K), "n_test": np.bincount(TE.label, minlength=K)}).to_string(index=False))
Y_TEST = TE.label.values
np.save(f"{OUT}/test_labels{'_smoke' if SMOKE else ''}.npy", Y_TEST)

# asosiy benchmark fayllari (ixtiyoriy Input)
MAIN_RES = glob.glob("/kaggle/input/**/results.jsonl", recursive=True)
MAIN_PREDS = {}
for f in glob.glob("/kaggle/input/**/*.npy", recursive=True):
    m = re.match(r"^uz_news_(.+)_(\d+)\.npy$", os.path.basename(f))
    if m: MAIN_PREDS[(m.group(1), int(m.group(2)))] = f
MAIN = None
if REUSE_BASELINE and MAIN_RES:
    MAIN = pd.read_json(io.StringIO(open(MAIN_RES[0]).read()), lines=True)
    MAIN = MAIN[MAIN.dataset == "uz_news"]
    print("asosiy benchmark topildi:", MAIN_RES[0], "| uz_news run'lar:", len(MAIN), "| preds:", len(MAIN_PREDS))
    # tiklangan test yorliqlari asosiy preds bilan mos kelishi shart
    for (m, s), f in MAIN_PREDS.items():
        r = MAIN[(MAIN.model == m) & (MAIN.seed == s)]
        if len(r) == 1:
            acc = float((np.load(f) == Y_TEST).mean())
            assert abs(acc - float(r.test_accuracy.iloc[0])) < 1e-6, f"{m} {s}: yorliqlar mos emas ({acc} != {float(r.test_accuracy.iloc[0])}). To'xtating."
    print("yorliq tekshiruvi: OK (qayta hisoblangan accuracy == asosiy benchmark)")
else:
    print("asosiy benchmark Input sifatida ulanmagan yoki SMOKE: c0 ham o'qitiladi.")

# %%
# ---------------- TOKENIZER / TRUNCATION DIAGNOSTIKASI (o'qitishsiz) ----------------
def load_tok(name):
    tok = AutoTokenizer.from_pretrained(name)
    try:
        tok(["a", "b"], is_split_into_words=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        tok(["a", "b"], is_split_into_words=True)
        print(f"[{name}] add_prefix_space=True bilan yuklandi")
    return tok

rows = []
sample = TE.text.tolist()[:1000]
for mk, mn in MODELS.items():
    tok = load_tok(mn)
    lens = np.array([len(x) for x in tok(sample, add_special_tokens=True, truncation=False)["input_ids"]])
    nwords = np.array([len(t.split()) for t in sample])
    kept = np.minimum(BASE["max_len"], lens) / lens
    unk = np.mean([sum(i == tok.unk_token_id for i in ids) / max(len(ids), 1) for ids in tok(sample, add_special_tokens=False)["input_ids"]])
    rows.append(dict(model=mk, tokens_per_doc=lens.mean(), docs_truncated=(lens > BASE["max_len"]).mean(), text_kept_fraction=kept.mean(), unk_rate=unk))
TRUNC = pd.DataFrame(rows); TRUNC.to_csv(f"{OUT}/truncation_diagnostics.csv", index=False)
print(f"max_len={BASE['max_len']} token bo'yicha qirqish (news test, birinchi {len(sample)} hujjat):\n", TRUNC.round(3).to_string(index=False))

# %%
# ---------------- TRAINING ----------------
_VALID_TA = {f.name for f in dataclasses.fields(TrainingArguments)}

def common_args(lr, epochs, bs, seed, outdir, n_train):
    total_steps = math.ceil(n_train / bs) * epochs
    kw = dict(output_dir=outdir, per_device_train_batch_size=bs, per_device_eval_batch_size=64, learning_rate=lr,
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
    f = (lambda t: norm_apos(t)) if cfg["cond"] == "norm_ascii" else (lambda t: t)
    def enc(df):
        e = dict(tok([f(t) for t in df.text], truncation=True, max_length=cfg["max_len"])); e["labels"] = df.label.tolist()
        return Dataset.from_dict(e)
    tr, va, te = enc(TR), enc(VA), enc(TE)
    model = AutoModelForSequenceClassification.from_pretrained(mname, num_labels=K)
    n_params = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model=model, args=common_args(cfg["lr"], cfg["epochs"], cfg["bs"], seed, outdir, len(tr)), train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok), processing_class=tok, compute_metrics=cls_metrics)
    t0 = time.time(); trainer.train(); train_s = time.time() - t0
    bv = trainer.evaluate(va); out = trainer.predict(te); pred = out.predictions.argmax(-1)
    pcf1, never = class_stats(Y_TEST, pred, K)
    best_epoch = None
    ck = getattr(trainer.state, "best_model_checkpoint", None)
    if ck and re.search(r"checkpoint-(\d+)", ck):
        best_epoch = round(int(re.search(r"checkpoint-(\d+)", ck).group(1)) / math.ceil(len(tr) / cfg["bs"]), 2)
    np.save(f"{OUT}/preds/{cfg_name}_{mkey}_{seed}{'_smoke' if SMOKE else ''}.npy", pred)
    rec = dict(cfg=cfg_name, model=mkey, seed=seed, lr=cfg["lr"], epochs=cfg["epochs"], cond=cfg["cond"], params=n_params,
               train_time_s=round(train_s, 1), peak_mem_mb=int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
               best_epoch=best_epoch, val_macro_f1=float(bv["eval_macro_f1"]), val_accuracy=float(bv["eval_accuracy"]),
               test_macro_f1=float(out.metrics["test_macro_f1"]), test_accuracy=float(out.metrics["test_accuracy"]),
               test_weighted_f1=float(out.metrics["test_weighted_f1"]), test_macro_precision=float(out.metrics["test_macro_precision"]),
               test_macro_recall=float(out.metrics["test_macro_recall"]), n_never_pred=never, per_class_f1=pcf1, source="trained")
    del trainer, model; shutil.rmtree(outdir, ignore_errors=True); gc.collect(); torch.cuda.empty_cache()
    return rec

def reused_baseline(mkey, seed):
    """c0 uchun asosiy benchmark yozuvini qaytaradi (bo'lsa)."""
    if MAIN is None: return None
    r = MAIN[(MAIN.model == mkey) & (MAIN.seed == seed)]
    if len(r) != 1: return None
    r = r.iloc[0]; pcf1, never = (None, None)
    if (mkey, seed) in MAIN_PREDS:
        pcf1, never = class_stats(Y_TEST, np.load(MAIN_PREDS[(mkey, seed)]), K)
    return dict(cfg="c0_base", model=mkey, seed=seed, lr=float(r.lr), epochs=int(r.epochs), cond="orig", params=int(r.params),
                train_time_s=float(r.train_time_s), peak_mem_mb=r.peak_mem_mb, best_epoch=None, val_macro_f1=float(r.val_macro_f1),
                val_accuracy=float(r.val_accuracy), test_macro_f1=float(r.test_macro_f1), test_accuracy=float(r.test_accuracy),
                test_weighted_f1=float(r.test_weighted_f1), test_macro_precision=float(r.test_macro_precision),
                test_macro_recall=float(r.test_macro_recall), n_never_pred=never, per_class_f1=pcf1, source="main_benchmark")

# %%
# ---------------- MAIN: pilot -> tanlash -> yakuniy ----------------
RES = f"{OUT}/news_tuned_results{'_smoke' if SMOKE else ''}.jsonl"
RECS = []
if os.path.exists(RES):
    RECS = [json.loads(l) for l in open(RES)]
print("Natijalar fayli:", RES, "| mavjud yozuvlar:", len(RECS))
T_START = time.time()

def ensure(cfg_name, mkey, seed):
    if any(r["cfg"] == cfg_name and r["model"] == mkey and r["seed"] == seed for r in RECS):
        return
    rec = reused_baseline(mkey, seed) if cfg_name == "c0_base" else None
    if rec is None:
        print(f"=== {cfg_name} | {mkey} | seed={seed} | o'tgan vaqt {(time.time() - T_START) / 3600:.2f} soat", flush=True)
        try:
            rec = run_cfg(cfg_name, mkey, seed)
        except Exception as e:
            open(f"{OUT}/errors.log", "a").write(f"{cfg_name} {mkey} {seed}: {repr(e)}\n"); print("XATO:", repr(e)[:300])
            shutil.rmtree(f"{OUT}/tmp_run", ignore_errors=True); gc.collect(); torch.cuda.empty_cache(); return
        print({k: (round(rec[k], 4) if isinstance(rec[k], float) else rec[k]) for k in ("val_macro_f1", "test_macro_f1", "test_accuracy", "n_never_pred", "best_epoch")})
    else:
        print(f"[qayta ishlatildi] {cfg_name} | {mkey} | seed={seed} (asosiy benchmark)")
    RECS.append(rec); open(RES, "a").write(json.dumps(rec, default=float) + "\n")

# 1) pilot
for cfg_name in CONFIGS:
    for mkey in MODELS:
        for seed in PILOT_SEEDS:
            ensure(cfg_name, mkey, seed)
PIL = pd.DataFrame([r for r in RECS if r["seed"] in PILOT_SEEDS])
SEL, AVG = select_configs(PIL, list(CONFIGS), PILOT_SEEDS)
AVG.pivot(index="model", columns="cfg", values="val_macro_f1").to_csv(f"{OUT}/pilot_val_macro_f1.csv")
json.dump(SEL, open(f"{OUT}/selected_configs.json", "w"), indent=2)
print("\nPilot: validatsiya Macro-F1 (model x konfiguratsiya):\n", AVG.pivot(index="model", columns="cfg", values="val_macro_f1").round(4).to_string())
print("Tanlangan konfiguratsiyalar (faqat validatsiya bo'yicha):", SEL)

# 2) yakuniy: tanlangan cfg, barcha seedlar (+ bazaviy c0, agar asosiy benchmarkdan bepul olinsa)
for mkey in MODELS:
    for seed in SEEDS:
        ensure(SEL[mkey], mkey, seed)
        if REUSE_BASELINE and MAIN is not None:
            ensure("c0_base", mkey, seed)
print(f"\nTugadi. Jami vaqt: {(time.time() - T_START) / 3600:.2f} soat")

# %%
# ---------------- AGGREGATION ----------------
def aggregate(R, SEL, names, train_cnt, out, seeds):
    """R: yozuvlar DataFrame. Yakuniy jadval = har model uchun tanlangan cfg (SEL) ning barcha seedlari."""
    fin = pd.concat([R[(R.model == m) & (R.cfg == c) & (R.seed.isin(seeds))] for m, c in SEL.items()])
    base = R[(R.cfg == "c0_base") & (R.seed.isin(seeds))]
    def summ(df, label):
        rows = []
        for m, g in df.groupby("model"):
            sd = lambda col: g[col].std(ddof=1) if len(g) > 1 else np.nan
            rows.append(dict(setting=label, model=m, cfg=g.cfg.iloc[0], n_seeds=len(g), test_macro_f1=g.test_macro_f1.mean(), std=sd("test_macro_f1"),
                             test_accuracy=g.test_accuracy.mean(), test_weighted_f1=g.test_weighted_f1.mean(),
                             seeds_with_collapse=(int((g.n_never_pred > 0).sum()) if g.n_never_pred.notna().all() else np.nan),
                             mean_never_pred=g.n_never_pred.mean() if g.n_never_pred.notna().all() else np.nan,
                             val_macro_f1=g.val_macro_f1.mean(), best_epoch=g.best_epoch.mean() if g.best_epoch.notna().any() else np.nan))
        return rows
    S = pd.DataFrame(summ(fin, "tuned") + (summ(base, "baseline") if len(base) else []))
    S.to_csv(f"{out}/news_fairness_summary.csv", index=False); print(S.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    # modellar orasidagi juft testlar (tuned): Welch + Holm
    pairs = []; ms = sorted(fin.model.unique())
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = fin[fin.model == ms[i]].test_macro_f1, fin[fin.model == ms[j]].test_macro_f1
            if len(a) > 1 and len(b) > 1:
                t, p = stats.ttest_ind(a, b, equal_var=False); pairs.append(dict(A=ms[i], B=ms[j], mean_diff=a.mean() - b.mean(), p=p))
    if pairs:
        P = pd.DataFrame(pairs); P["p_holm"] = holm(P.p.values); P.to_csv(f"{out}/news_fairness_tests_tuned.csv", index=False)
        print("\nModellar orasida (tuned, Welch + Holm):\n", P.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    # bazaviy va tuned (har model)
    if len(base):
        rows = []
        for m in ms:
            a, b = base[base.model == m].test_macro_f1, fin[fin.model == m].test_macro_f1
            if len(a) > 1 and len(b) > 1 and SEL[m] != "c0_base":
                t, p = stats.ttest_ind(b, a, equal_var=False); rows.append(dict(model=m, selected=SEL[m], mean_gain=b.mean() - a.mean(), p=p))
        if rows:
            G = pd.DataFrame(rows); G["p_holm"] = holm(G.p.values); G.to_csv(f"{out}/news_fairness_tuned_vs_baseline.csv", index=False)
            print("\nTuned va baseline (Welch + Holm):\n", G.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    # sinf bo'yicha F1 (tuned) va kam uchraydigan sinflar
    pc = fin[fin.per_class_f1.notna()]
    if len(pc):
        tab = pd.DataFrame({"class": names, "n_train": train_cnt})
        for m, g in pc.groupby("model"):
            arr = np.array(g.per_class_f1.tolist()); tab[f"{m}_f1_mean"] = arr.mean(0)
        tab.to_csv(f"{out}/news_fairness_per_class_f1.csv", index=False)
        rare = tab.n_train <= 110
        print(f"\nKam uchraydigan sinflar (n_train<=110): {tab['class'][rare].tolist()}")
        print("kam sinflar o'rtacha F1:", {c.replace('_f1_mean', ''): round(tab.loc[rare, c].mean(), 3) for c in tab.columns if c.endswith("_f1_mean")})
        print(tab.round(3).to_string(index=False))
    # grafik
    fig, ax = plt.subplots(figsize=(9, 4.5)); x = np.arange(len(ms)); w = 0.38
    for k, (lab, df) in enumerate([("baseline (c0)", base), ("tuned", fin)]):
        if len(df):
            s = df.groupby("model").test_macro_f1.agg(["mean", "std"]).reindex(ms)
            ax.bar(x + k * w, s["mean"], w, yerr=s["std"], capsize=2, label=lab)
    ax.set_xticks(x + w / 2); ax.set_xticklabels(ms); ax.set_ylabel("news test Macro-F1"); ax.legend(); plt.tight_layout()
    plt.savefig(f"{out}/news_fairness_macro_f1.png", dpi=130); plt.close()
    return S

RDF = pd.DataFrame(RECS)
assert len(RDF) > 0, "natijalar yo'q: errors.log ni tekshiring"
SUM = aggregate(RDF, SEL, NAMES, TRAIN_CNT, OUT, SEEDS)
print("\nTAYYOR. Natijalar:", OUT)
