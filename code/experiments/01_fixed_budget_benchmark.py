# %% [markdown]
# # Uzbek encoder benchmark: text classification + NER
# Models: XLM-R, mBERT, TahrirchiBERT (+ optional). Tasks: SIB-200 (uzn_Latn), WikiANN (uz), Uzbek news classification.
# Kaggle sozlamalari: Accelerator = GPU P100 (yoki T4 x1), Internet = On, Persistence = Files.
# Avval SMOKE = True bilan butun pipeline'ni tekshiring, keyin SMOKE = False.

# %%
import os, sys, json, random, time, gc, shutil, platform, subprocess
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # T4 x2 tanlangan bo'lsa ham bitta GPU: batch size aniq 16/32 bo'ladi (DataParallel yo'q)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np, pandas as pd, torch, transformers, datasets, sklearn
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding,
                          DataCollatorForTokenClassification, set_seed)
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

def get_entities(seq):
    """IOB2 ketma-ketlikdan (tur, boshlanish, tugash) to'plamini ajratadi."""
    ents, start, typ = set(), None, None
    for i, tag in enumerate(list(seq) + ["O"]):
        if tag == "O" or tag.startswith("B-") or tag[2:] != typ:
            if typ is not None:
                ents.add((typ, start, i))
            if tag == "O":
                start, typ = None, None
            else:
                start, typ = i, tag[2:]
    return ents

def entity_prf(y_true, y_pred):
    tp = n_pred = n_true = 0
    for t, p in zip(y_true, y_pred):
        te, pe = get_entities(t), get_entities(p)
        tp += len(te & pe); n_pred += len(pe); n_true += len(te)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_true if n_true else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

# kichik o'z-o'zini tekshirish
assert entity_prf([["B-PER", "I-PER", "O", "B-LOC"]], [["B-PER", "I-PER", "O", "O"]]) == (1.0, 0.5, 2 / 3)

def load_tok(name):
    """Tokenizer yuklaydi. RoBERTa tipidagi byte-level BPE tokenizerlar (masalan TahrirchiBERT) so'zlarga bo'lingan kirish
    (is_split_into_words=True) uchun add_prefix_space=True talab qiladi; boshqa tokenizerlar o'zgarishsiz qoladi."""
    tok = AutoTokenizer.from_pretrained(name)
    try:
        tok(["a", "b"], is_split_into_words=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        tok(["a", "b"], is_split_into_words=True)    # yana xato bersa, shu yerda aniq to'xtaydi
        print(f"[{name}] add_prefix_space=True bilan yuklandi")
    return tok

ENV = {
    "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
    "datasets": datasets.__version__, "sklearn": sklearn.__version__,
    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}
print(ENV)
OUT = "/kaggle/working"
os.makedirs(f"{OUT}/preds", exist_ok=True)
json.dump(ENV, open(f"{OUT}/environment.json", "w"), indent=2)

# %%
# ---------------- CONFIG ----------------
SMOKE = True                      # True: 1 seed, 1 epoch, kichik subsetlar (faqat tekshiruv uchun)
SEEDS = [13, 21, 42, 87, 100]
MODELS = {
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "tahrirchi-base": "tahrirchi/tahrirchi-bert-base",
    "tahrirchi-small": "tahrirchi/tahrirchi-bert-small",
    # "uzbert": "coppercitylabs/uzbert-base-uncased",  # yozuvi (Lotin/Kirill) ni tekshiring, keyin yoqing
}
CFG = {  # barcha modellar uchun bir xil (controlled setting)
    "sib200": dict(task="cls", lr=3e-5, epochs=10, bs=16, max_len=128),
    "wikiann_uz": dict(task="ner", lr=3e-5, epochs=10, bs=16, max_len=128),
    "uz_news": dict(task="cls", lr=2e-5, epochs=3, bs=32, max_len=256, n_train=10000, n_val=2000, n_test=5000),
}
NEWS_TEXT_COL, NEWS_LABEL_COL = None, None   # None = avtomatik aniqlanadi; noto'g'ri bo'lsa qo'lda yozing
if SMOKE:
    SEEDS = SEEDS[:1]
    for c in CFG.values():
        c["epochs"] = 1
    CFG["uz_news"].update(n_train=500, n_val=200, n_test=200)
    MODELS = dict(list(MODELS.items())[:2])

# %%
# ---------------- DATA ----------------
def load_sib():
    try:
        d = load_dataset("Davlan/sib200", "uzn_Latn")
        tr, va, te = [pd.DataFrame(d[s]) for s in ("train", "validation", "test")]
    except Exception as e:
        print("HF yuklash xatosi, GitHub zaxira yo'lidan foydalaniladi:", repr(e)[:200])
        base = "https://github.com/dadelani/sib-200/raw/main/data/annotated/uzn_Latn/"
        tr, va, te = [pd.read_csv(base + f"{s}.tsv", sep="\t") for s in ("train", "dev", "test")]
    lab = "category"
    labels = sorted(tr[lab].unique())
    f = lambda df: pd.DataFrame({"text": df["text"], "label": df[lab]})
    return dict(train=f(tr), val=f(va), test=f(te), labels=labels)

def load_wikiann():
    d = load_dataset("unimelb-nlp/wikiann", "uz")
    names = d["train"].features["ner_tags"].feature.names
    return dict(train=d["train"], val=d["validation"], test=d["test"], labels=names)

def load_news(cfg):
    d = load_dataset("murodbek/uz-text-classification")
    cols = d["train"].column_names
    print("news columns:", cols); print("example:", {k: str(v)[:80] for k, v in d["train"][0].items()})
    tcol = NEWS_TEXT_COL or next(c for c in ("text", "content", "article") if c in cols)
    lcol = NEWS_LABEL_COL or next(c for c in ("label", "category", "labels") if c in cols)
    feat = d["train"].features[lcol]
    names = getattr(feat, "names", None)
    def prep(split, n):
        s = d[split].shuffle(seed=0).select(range(min(n, len(d[split]))))   # subsample sobit (seed=0), barcha model uchun bir xil
        df = pd.DataFrame({"text": s[tcol], "label": s[lcol]})
        if names: df["label"] = df["label"].map(lambda i: names[i])
        assert df["label"].map(lambda x: isinstance(x, (str, int))).all(), "Label ko'p-label (list) ko'rinishida: NEWS_LABEL_COL ni tekshiring"
        df["label"] = df["label"].astype(str)
        return df
    tr, va, te = prep("train", cfg["n_train"]), prep("validation", cfg["n_val"]), prep("test", cfg["n_test"])
    # sinflar ro'yxati: butun datasetdagi barcha sinflar (kichik subsample'da ba'zi sinflar train'da bo'lmasligi mumkin)
    labels = [str(n) for n in names] if names else sorted(set(tr["label"]) | set(va["label"]) | set(te["label"]))
    print("news class distribution (train):\n", tr["label"].value_counts())
    print("news class distribution (test):\n", te["label"].value_counts())
    return dict(train=tr, val=va, test=te, labels=labels)

DATA = {"sib200": load_sib(), "wikiann_uz": load_wikiann(), "uz_news": load_news(CFG["uz_news"])}
for k, v in DATA.items():
    print(k, {s: len(v[s]) for s in ("train", "val", "test")}, "classes:", len(v["labels"]))

# %%
# ---------------- TOKENIZER ANALYSIS (fertility) ----------------
def words_of(name, split="train", limit=2000):
    d = DATA[name][split]
    if CFG[name]["task"] == "ner":
        return [list(t) for t in d["tokens"][:limit]]
    return [t.split() for t in d["text"].tolist()[:limit]]

def tok_stats(tok, wl):
    enc = tok(wl, is_split_into_words=True, add_special_tokens=False)
    n_words = n_tokens = n_split = n_unk = 0
    for i, ws in enumerate(wl):
        ids = enc["input_ids"][i]; wid = enc.word_ids(i)
        counts = {}
        for w in wid:
            counts[w] = counts.get(w, 0) + 1
        n_words += len(ws); n_tokens += len(ids)
        n_split += sum(1 for c in counts.values() if c > 1)
        n_unk += sum(1 for t in ids if t == tok.unk_token_id)
    return dict(fertility=n_tokens / n_words, split_word_rate=n_split / n_words, unk_rate=n_unk / n_tokens)

rows = []
for mk, mn in MODELS.items():
    tok = load_tok(mn)
    assert tok.is_fast, f"{mn}: fast tokenizer kerak (word_ids uchun)"
    for dn in DATA:
        rows.append(dict(model=mk, dataset=dn, **tok_stats(tok, words_of(dn))))
FERT = pd.DataFrame(rows); FERT.to_csv(f"{OUT}/tokenizer_stats.csv", index=False)
print(FERT.pivot(index="model", columns="dataset", values="fertility"))

# %%
# ---------------- TRAINING ----------------
import dataclasses, math
_VALID_TA = {f.name for f in dataclasses.fields(TrainingArguments)}

def common_args(cfg, seed, outdir, n_train):
    # warmup_ratio yangi transformers'da olib tashlangan -> 10% ni qadamlar soniga aylantiramiz (versiyadan mustaqil)
    total_steps = math.ceil(n_train / cfg["bs"]) * cfg["epochs"]
    kw = dict(
        output_dir=outdir, per_device_train_batch_size=cfg["bs"], per_device_eval_batch_size=64,
        learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"], weight_decay=0.01,
        warmup_steps=int(0.1 * total_steps),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="macro_f1" if cfg["task"] == "cls" else "f1", greater_is_better=True,
        fp16=torch.cuda.is_available(), seed=seed, data_seed=seed, report_to="none", logging_steps=100,
        dataloader_num_workers=2, disable_tqdm=True)
    dropped = [k for k in kw if k not in _VALID_TA]
    if dropped:
        print("OGOHLANTIRISH: TrainingArguments quyidagilarni qo'llamaydi va tashlab yuborildi:", dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in _VALID_TA})

def cls_metrics(p):
    pred = p.predictions.argmax(-1); y = p.label_ids
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "weighted_f1": f1_score(y, pred, average="weighted"),
            "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y, pred, average="macro", zero_division=0)}

def make_ner_metrics(names):
    def f(p):
        pred = p.predictions.argmax(-1); yt, yp = [], []
        for pr, lb in zip(pred, p.label_ids):
            yt.append([names[l] for l in lb if l != -100]); yp.append([names[x] for x, l in zip(pr, lb) if l != -100])
        pr_, rc_, f1_ = entity_prf(yt, yp)
        return {"f1": f1_, "precision": pr_, "recall": rc_}
    return f

def run_one(ds_name, mkey, seed):
    cfg, data, mname = CFG[ds_name], DATA[ds_name], MODELS[mkey]
    set_seed(seed)
    tok = load_tok(mname)
    labels = data["labels"]; outdir = f"{OUT}/tmp_run"
    if cfg["task"] == "cls":
        l2i = {l: i for i, l in enumerate(labels)}
        def enc(df):
            e = dict(tok(df["text"].tolist(), truncation=True, max_length=cfg["max_len"]))
            e["labels"] = [l2i[x] for x in df["label"]]
            return Dataset.from_dict(e)
        tr, va, te = enc(data["train"]), enc(data["val"]), enc(data["test"])
        model = AutoModelForSequenceClassification.from_pretrained(mname, num_labels=len(labels))
        collator, metrics = DataCollatorWithPadding(tok), cls_metrics
    else:
        def tk(batch):
            e = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=cfg["max_len"])
            out = []
            for i, tags in enumerate(batch["ner_tags"]):
                prev, lab = None, []
                for w in e.word_ids(batch_index=i):
                    lab.append(-100 if (w is None or w == prev) else tags[w]); prev = w
                out.append(lab)
            r = dict(e); r["labels"] = out; return r
        tr, va, te = [data[s].map(tk, batched=True, remove_columns=data[s].column_names) for s in ("train", "val", "test")]
        model = AutoModelForTokenClassification.from_pretrained(
            mname, num_labels=len(labels), id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)})
        collator, metrics = DataCollatorForTokenClassification(tok), make_ner_metrics(labels)
    n_params = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model=model, args=common_args(cfg, seed, outdir, len(tr)), train_dataset=tr, eval_dataset=va,
                      data_collator=collator, processing_class=tok, compute_metrics=metrics)
    t0 = time.time(); trainer.train(); train_s = time.time() - t0
    best_val = trainer.evaluate(va)
    t0 = time.time(); out = trainer.predict(te); infer_s = time.time() - t0
    rec = dict(dataset=ds_name, model=mkey, seed=seed, params=n_params, train_time_s=round(train_s, 1),
               test_time_s=round(infer_s, 2), peak_mem_mb=int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
               **{k: v for k, v in cfg.items() if k in ("lr", "epochs", "bs", "max_len")},
               **{k.replace("eval_", "val_"): v for k, v in best_val.items() if isinstance(v, float)},
               **{k.replace("test_", "test_"): v for k, v in out.metrics.items() if k.startswith("test_") and isinstance(v, float)})
    if cfg["task"] == "cls":
        np.save(f"{OUT}/preds/{ds_name}_{mkey}_{seed}.npy", out.predictions.argmax(-1))
    del trainer, model; shutil.rmtree(outdir, ignore_errors=True); gc.collect(); torch.cuda.empty_cache()
    return rec

# %%
# ---------------- MAIN LOOP (resume-safe) ----------------
RES = f"{OUT}/results_smoke.jsonl" if SMOKE else f"{OUT}/results.jsonl"   # smoke natijalari asosiy natijalarga aralashmasin
print("Natijalar fayli:", RES)
done = set()
if os.path.exists(RES):
    for line in open(RES):
        r = json.loads(line); done.add((r["dataset"], r["model"], r["seed"]))
for ds_name in DATA:
    for mkey in MODELS:
        for seed in SEEDS:
            if (ds_name, mkey, seed) in done: continue
            print(f"=== {ds_name} | {mkey} | seed={seed}", flush=True)
            try:
                rec = run_one(ds_name, mkey, seed)
                open(RES, "a").write(json.dumps(rec, default=float) + "\n"); print({k: rec[k] for k in rec if k.startswith("test_")})
            except Exception as e:
                open(f"{OUT}/errors.log", "a").write(f"{ds_name} {mkey} {seed}: {repr(e)}\n"); print("XATO:", repr(e)[:300])
                shutil.rmtree(f"{OUT}/tmp_run", ignore_errors=True); gc.collect(); torch.cuda.empty_cache()

# %%
# ---------------- AGGREGATION + STATISTICS ----------------
from scipy import stats
import io
assert os.path.exists(RES) and os.path.getsize(RES) > 0, "results.jsonl bo'sh: avval main loop cell'ini xatosiz bajaring (errors.log ni ko'ring)"
R = pd.read_json(io.StringIO(open(RES).read()), lines=True)
main = {"sib200": "test_macro_f1", "uz_news": "test_macro_f1", "wikiann_uz": "test_f1"}
summ = []
for dn, col in main.items():
    sub = R[R.dataset == dn]
    for mk, g in sub.groupby("model"):
        v = g[col].values; n = len(v)
        ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        summ.append(dict(dataset=dn, model=mk, n_seeds=n, mean=v.mean(), std=v.std(ddof=1) if n > 1 else np.nan, ci95=ci))
SUMM = pd.DataFrame(summ); SUMM.to_csv(f"{OUT}/summary.csv", index=False); print(SUMM.round(4))

# Welch t-test (modellar har xil, seedlar juftlashmagan) + Holm tuzatishi
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj
tests = []
for dn, col in main.items():
    sub = R[R.dataset == dn]; ms = sorted(sub.model.unique()); pairs = []
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = sub[sub.model == ms[i]][col], sub[sub.model == ms[j]][col]
            if len(a) > 1 and len(b) > 1:
                t, p = stats.ttest_ind(a, b, equal_var=False); pairs.append((dn, ms[i], ms[j], a.mean() - b.mean(), p))
    if pairs:
        adj = holm([x[4] for x in pairs])
        tests += [dict(dataset=x[0], A=x[1], B=x[2], mean_diff=x[3], p=x[4], p_holm=a) for x, a in zip(pairs, adj)]
TESTS = pd.DataFrame(tests); TESTS.to_csv(f"{OUT}/stat_tests.csv", index=False); print(TESTS.round(4))

# Fertility va sifat orasidagi bog'liqlik (n kichik: Spearman faqat tavsifiy)
M = SUMM.merge(FERT, on=["dataset", "model"])
for dn, g in M.groupby("dataset"):
    if len(g) >= 4:
        rho, p = stats.spearmanr(g["fertility"], g["mean"]); print(f"{dn}: Spearman(fertility, score) = {rho:.3f} (p={p:.3f}, n={len(g)})")
M.to_csv(f"{OUT}/fertility_vs_score.csv", index=False)
