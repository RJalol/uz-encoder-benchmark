# %% [markdown]
# # Uzbek encoders: error analysis (A) va NER ablation/diagnostika (B)
# **Oldin:** asosiy benchmark notebook'i (60 run) tugagan bo'lishi kerak.
#
# **A qism** (o'qitishsiz): klassifikatsiya xatolarini tahlil qiladi: sinf bo'yicha F1, confusion, uzunlik, qiyin misollar.
# Buning uchun asosiy run natijalari (`preds/*.npy` va `results.jsonl`) Kaggle **Input** sifatida ulangan bo'lishi kerak
# (Add data -> Upload -> results zip faylini yuklang).
#
# **B qism** (GPU kerak): WikiANN-uz uchun (1) tokenizer diagnostikasi, (2) apostrof normalizatsiyasi ablation'i,
# (3) xato turlari va entity turi (PER/ORG/LOC) bo'yicha tahlil. Ixtiyoriy: o'rganish tezligi (LR) tekshiruvi.
#
# Kaggle sozlamalari: Internet = On, Accelerator = GPU (P100 yoki T4 x1). Avval `SMOKE = True` bilan tekshiring.

# %%
import os, sys, json, glob, time, gc, shutil, math, platform, unicodedata, re
from collections import Counter
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import torch, transformers, datasets
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer,
                          DataCollatorForTokenClassification, set_seed)

OUT = "/kaggle/working/analysis"
os.makedirs(OUT, exist_ok=True)
ENV = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
       "datasets": datasets.__version__, "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
print(ENV)
json.dump(ENV, open(f"{OUT}/environment_analysis.json", "w"), indent=2)

# %%
# ---------------- CONFIG ----------------
SMOKE = True            # True: tezkor tekshiruv (1 seed, 1 epoch, kichik subset). To'liq run uchun False.
RUN_PART_A = True       # klassifikatsiya error analysis (GPU kerak emas)
RUN_PART_B = True       # NER ablation (GPU kerak)

SEEDS = [13, 21, 42, 87, 100]                  # asosiy benchmark bilan bir xil
MODELS = {
    "xlmr-base": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
    "tahrirchi-base": "tahrirchi/tahrirchi-bert-base",
    "tahrirchi-small": "tahrirchi/tahrirchi-bert-small",
}
# B qism shartlari: "orig" = asosiy benchmarkdagi kabi (nazorat). "norm_ascii" = barcha apostrof-o'xshash belgilar ASCII ' ga.
# "norm_u2018" = barchasi U+2018 (‘) ga (TahrirchiBERT model kartasi shu belgini ishlatadi).
CONDITIONS = ["orig", "norm_ascii"]
LRS = [3e-5]            # 1-bosqich: faqat 3e-5 (asosiy benchmarkdagi). 2-bosqich uchun: [5e-5, 1e-4] ga o'zgartiring va CONDITIONS = ["orig"]
NER_CFG = dict(epochs=10, bs=16, max_len=128)
if SMOKE:
    SEEDS = SEEDS[:1]
    NER_CFG["epochs"] = 1
PRED_DIR_B = f"{OUT}/ner_preds{'_smoke' if SMOKE else ''}"   # smoke va to'liq run bashoratlari aralashmasin

# %%
# ---------------- PURE FUNCTIONS (torch'siz; alohida sinalgan) ----------------
APOS_CHARS = {"’": "RIGHT SINGLE QUOTATION MARK", "‘": "LEFT SINGLE QUOTATION MARK",
              "ʼ": "MODIFIER LETTER APOSTROPHE", "ʻ": "MODIFIER LETTER TURNED COMMA",
              "´": "ACUTE ACCENT", "`": "GRAVE ACCENT", "′": "PRIME", "ʹ": "MODIFIER LETTER PRIME"}
NORM_TARGETS = {"norm_ascii": "'", "norm_u2018": "‘"}

def norm_apos(s, target):
    s = unicodedata.normalize("NFC", s)
    for c in APOS_CHARS:
        s = s.replace(c, target)
    if target != "'":
        s = s.replace("'", target)
    return s

def has_apos(s):
    return any(c in s for c in list(APOS_CHARS) + ["'"])

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

def per_type_prf(y_true, y_pred, types=("PER", "ORG", "LOC")):
    tp, npred, ngold = Counter(), Counter(), Counter()
    for t, p in zip(y_true, y_pred):
        G, P = get_entities(t), get_entities(p)
        for e in G: ngold[e[0]] += 1
        for e in P: npred[e[0]] += 1
        for e in G & P: tp[e[0]] += 1
    rows = {}
    for ty in types:
        pr = tp[ty] / npred[ty] if npred[ty] else 0.0
        rc = tp[ty] / ngold[ty] if ngold[ty] else 0.0
        rows[ty] = dict(precision=pr, recall=rc, f1=(2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0,
                        n_gold=ngold[ty], n_pred=npred[ty])
    return rows

def error_taxonomy(y_true, y_pred):
    """Gold entity bo'yicha: correct / missed / type_only (span to'g'ri, tur xato) /
    boundary_only (tur to'g'ri, chegara xato) / boundary_and_type; pred bo'yicha: spurious (hech qanday gold bilan kesishmaydi)."""
    c = Counter()
    for t, p in zip(y_true, y_pred):
        G, P = get_entities(t), get_entities(p)
        for g in G:
            if g in P:
                c["correct"] += 1; continue
            ov = [q for q in P if q[1] < g[2] and g[1] < q[2]]
            if not ov: c["missed"] += 1
            elif any(q[1] == g[1] and q[2] == g[2] for q in ov): c["type_only"] += 1
            elif any(q[0] == g[0] for q in ov): c["boundary_only"] += 1
            else: c["boundary_and_type"] += 1
        for q in P:
            if not any(q[1] < g[2] and g[1] < q[2] for g in G):
                c["spurious"] += 1
    return c

# kichik o'z-o'zini tekshirishlar
assert entity_prf([["B-PER", "I-PER", "O", "B-LOC"]], [["B-PER", "I-PER", "O", "O"]]) == (1.0, 0.5, 2 / 3)
assert norm_apos("o‘zbek gʻalaba ma’no", "'") == "o'zbek g'alaba ma'no"
assert norm_apos("o'zbek", "‘") == "o‘zbek"
_c = error_taxonomy([["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG"]], [["B-PER", "O", "O", "B-ORG", "O", "O"]])
assert _c["boundary_only"] == 1 and _c["type_only"] == 1 and _c["missed"] == 1 and _c["spurious"] == 0, _c
print("pure-function self-tests OK")

# %%
# ======================= A QISM: klassifikatsiya error analysis =======================
def find_inputs():
    preds = {}
    for f in glob.glob("/kaggle/input/**/*.npy", recursive=True):
        m = re.match(r"^(sib200|uz_news)_(.+)_(\d+)\.npy$", os.path.basename(f))
        if m:
            preds[(m.group(1), m.group(2), int(m.group(3)))] = f
    res = glob.glob("/kaggle/input/**/results.jsonl", recursive=True)
    return preds, (res[0] if res else None)

def load_cls_tests():
    out = {}
    d = load_dataset("Davlan/sib200", "uzn_Latn")
    trd, ted = pd.DataFrame(d["train"]), pd.DataFrame(d["test"])
    labels = sorted(trd["category"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    out["sib200"] = dict(labels=labels, y=ted["category"].map(l2i).values, text=ted["text"].tolist(),
                         train_y=trd["category"].map(l2i).values)
    n = load_dataset("murodbek/uz-text-classification")
    names = [str(x) for x in n["train"].features["label"].names]
    tr = n["train"].shuffle(seed=0).select(range(10000))       # asosiy benchmarkdagi subsample bilan bir xil
    te = n["test"].shuffle(seed=0).select(range(5000))
    out["uz_news"] = dict(labels=names, y=np.array(te["label"]), text=list(te["text"]), train_y=np.array(tr["label"]))
    return out

if RUN_PART_A:
    PRED_FILES, RES_FILE = find_inputs()
    print("topilgan pred fayllari:", len(PRED_FILES), "| results.jsonl:", RES_FILE)
    if not PRED_FILES:
        print("OGOHLANTIRISH: /kaggle/input ichida preds/*.npy topilmadi. A qism o'tkazib yuboriladi.\n"
              "Add data -> Upload orqali asosiy run natijalari zip'ini yuklab, notebook'ga ulang.")
        RUN_PART_A = False

def analyze_cls(ds_name, info, P, REC, out_dir, show=True):
    """info: labels, y, text, train_y.  P: {(model, seed): pred_indices}.  REC: results.jsonl DataFrame yoki None."""
    labels, y, K = info["labels"], np.asarray(info["y"]), len(info["labels"])
    models = sorted({m for (m, _) in P}); seeds_of = {m: sorted(s for (mm, s) in P if mm == m) for m in models}
    print(f"\n=== {ds_name}: {len(y)} test, {K} sinf, modellar: {models}")
    # --- tekshiruv: tiklangan yorliqlar va pred tartibi asosiy run bilan mos kelishi shart
    bad = checked = 0
    for (m, s), p in P.items():
        assert len(p) == len(y), f"{ds_name} {m} {s}: pred uzunligi {len(p)} != test {len(y)}"
        if REC is not None:
            r = REC[(REC.dataset == ds_name) & (REC.model == m) & (REC.seed == s)]
            if len(r) == 1:
                checked += 1
                if abs(float((p == y).mean()) - float(r.test_accuracy.iloc[0])) > 1e-6:
                    bad += 1
    print(f"accuracy tekshiruvi (qayta hisoblangan == results.jsonl): {checked} ta run tekshirildi, mos kelmaganlari: {bad}"
          if REC is not None else "results.jsonl ulanmagan: accuracy tekshiruvi O'TKAZILMADI")
    assert bad == 0, "Yorliq/tartib mos emas: shuffle(seed=0) yoki dataset versiyasi boshqacha. To'xtating va menga yuboring."

    # --- sinf bo'yicha F1 (seed bo'yicha o'rtacha/std)
    f1 = {m: np.array([precision_recall_fscore_support(y, P[(m, s)], labels=list(range(K)), zero_division=0)[2] for s in seeds_of[m]]) for m in models}
    tr_cnt = np.bincount(info["train_y"], minlength=K); te_cnt = np.bincount(y, minlength=K)
    tab = pd.DataFrame({"class": labels, "n_train": tr_cnt, "n_test": te_cnt})
    for m in models:
        tab[f"{m}_f1_mean"] = f1[m].mean(0); tab[f"{m}_f1_std"] = f1[m].std(0, ddof=1) if len(f1[m]) > 1 else np.nan
    tab.to_csv(f"{out_dir}/A_{ds_name}_per_class_f1.csv", index=False)
    print(tab[["class", "n_train", "n_test"] + [f"{m}_f1_mean" for m in models]].round(3).to_string(index=False))
    for m in models:
        rho, p = stats.spearmanr(tr_cnt, f1[m].mean(0)); print(f"  Spearman(train support, class F1) {m}: rho={rho:.2f} (n={K} sinf, p={p:.3f})")

    # --- confusion (barcha seedlar yig'indisi) + eng ko'p chalkashliklar
    nrow = int(np.ceil(len(models) / 2)); fig, axes = plt.subplots(nrow, 2, figsize=(14, 6 * nrow), squeeze=False); conf_rows = []
    for ax, m in zip(axes.ravel(), models):
        cm = sum(confusion_matrix(y, P[(m, s)], labels=list(range(K))) for s in seeds_of[m]); nseed = len(seeds_of[m])
        ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), vmin=0, vmax=1, cmap="Blues"); ax.set_title(m)
        ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(labels, rotation=90, fontsize=7); ax.set_yticklabels(labels, fontsize=7)
        off = [(cm[i, j] / nseed, labels[i], labels[j]) for i in range(K) for j in range(K) if i != j and cm[i, j] > 0]
        for cnt, gi, pj in sorted(off, reverse=True)[:8]:
            conf_rows.append(dict(model=m, gold=gi, predicted=pj, avg_count_per_seed=round(cnt, 2)))
    plt.tight_layout(); plt.savefig(f"{out_dir}/A_{ds_name}_confusion.png", dpi=130); plt.close()
    CR = pd.DataFrame(conf_rows); CR.to_csv(f"{out_dir}/A_{ds_name}_top_confusions.csv", index=False)
    print(CR.groupby("model").head(4).to_string(index=False))

    # --- matn uzunligi bo'yicha accuracy (kvartillar)
    L = pd.Series([len(t) for t in info["text"]])
    bins = pd.qcut(L, 4, labels=["Q1 (eng qisqa)", "Q2", "Q3", "Q4 (eng uzun)"], duplicates="drop")
    len_rows = []
    for m in models:
        acc_seed = np.array([(P[(m, s)] == y) for s in seeds_of[m]])
        for bname in bins.cat.categories:
            idx = (bins == bname).values
            len_rows.append(dict(model=m, length_bin=bname, n=int(idx.sum()), accuracy=acc_seed[:, idx].mean()))
    LR_ = pd.DataFrame(len_rows); LR_.to_csv(f"{out_dir}/A_{ds_name}_accuracy_by_length.csv", index=False)
    print(LR_.pivot(index="length_bin", columns="model", values="accuracy").round(3))

    # --- qiyin misollar: barcha run'larning >=90% ida xato
    err = np.mean([(P[k] != y) for k in P], axis=0); hard = np.where(err >= 0.9)[0]
    print(f"barcha modellar/seedlar bo'yicha >=90% xato qilingan misollar: {len(hard)} / {len(y)}")
    rows = []
    for i in hard[:50]:
        maj = Counter(int(P[k][i]) for k in P).most_common(1)[0][0]
        rows.append(dict(idx=int(i), gold=labels[int(y[i])], majority_pred=labels[maj], text=info["text"][i][:300]))
    pd.DataFrame(rows).to_csv(f"{out_dir}/A_{ds_name}_hard_examples.csv", index=False)
    for r in rows[:5]:
        print(f"  [{r['gold']} -> {r['majority_pred']}] {r['text'][:140]}")
    return tab

if RUN_PART_A:
    CLS = load_cls_tests()
    import io
    REC = pd.read_json(io.StringIO(open(RES_FILE).read()), lines=True) if RES_FILE else None
    for ds_name, info in CLS.items():
        P = {(m, s): np.load(PRED_FILES[(d, m, s)]) for (d, m, s) in PRED_FILES if d == ds_name}
        analyze_cls(ds_name, info, P, REC, OUT)

# %%
# ======================= B QISM: NER diagnostika =======================
def load_wikiann(smoke):
    d = load_dataset("unimelb-nlp/wikiann", "uz")
    names = d["train"].features["ner_tags"].feature.names
    tr, va, te = d["train"], d["validation"], d["test"]
    if smoke:
        tr, va, te = tr.select(range(200)), va.select(range(100)), te.select(range(200))
    return dict(train=tr, val=va, test=te, labels=names)

def load_tok(name):
    tok = AutoTokenizer.from_pretrained(name)
    try:
        tok(["a", "b"], is_split_into_words=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained(name, add_prefix_space=True)
        tok(["a", "b"], is_split_into_words=True)
        print(f"[{name}] add_prefix_space=True bilan yuklandi")
    return tok

def word_token_counts(tok, words_list):
    enc = tok(words_list, is_split_into_words=True, add_special_tokens=False)
    out = []
    for i, ws in enumerate(words_list):
        c = Counter(w for w in enc.word_ids(i) if w is not None)
        out.append([c.get(j, 0) for j in range(len(ws))])
    return out

if RUN_PART_B:
    NER = load_wikiann(SMOKE)
    NAMES = NER["labels"]; print("NER teglar:", NAMES, {k: len(NER[k]) for k in ("train", "val", "test")})

    # --- B1a: apostrof-o'xshash belgilar ro'yxati (WikiANN train)
    census = Counter(ch for toks in NER["train"]["tokens"] for t in toks for ch in t if ch in APOS_CHARS or ch == "'")
    cen = pd.DataFrame([dict(char=c, codepoint=f"U+{ord(c):04X}", name=APOS_CHARS.get(c, "APOSTROPHE (ASCII)"), count=n) for c, n in census.most_common()])
    cen.to_csv(f"{OUT}/B_apostrophe_census_wikiann_train.csv", index=False); print("\nWikiANN-uz train apostrof belgilari:\n", cen.to_string(index=False) if len(cen) else "(topilmadi)")
    n_words = sum(len(t) for t in NER["train"]["tokens"])
    n_apos_words = sum(has_apos(w) for t in NER["train"]["tokens"] for w in t)
    print(f"apostrofli so'zlar ulushi: {n_apos_words}/{n_words} = {n_apos_words / n_words:.4f}")

    # --- B1b: tokenizer diagnostikasi (apostrof variantlari; entity va entity bo'lmagan so'zlar)
    PROBES = ["o'zbek", "oʻzbek", "o‘zbek", "o’zbek", "g'alaba", "gʻalaba", "g‘alaba", "g’alaba", "ma'no", "ma’no"]
    probe_rows, diag_rows = [], []
    sub = NER["train"].select(range(min(1000, len(NER["train"]))))
    words_list = [list(t) for t in sub["tokens"]]; tags_list = [list(t) for t in sub["ner_tags"]]
    for mk, mn in MODELS.items():
        tok = load_tok(mn); assert tok.is_fast
        for w in PROBES:
            ids = tok([w], is_split_into_words=True, add_special_tokens=False)["input_ids"]   # bitta misol -> tekis ro'yxat
            probe_rows.append(dict(model=mk, word=w, n_tokens=len(ids), tokens=" ".join(tok.convert_ids_to_tokens(ids))))
        counts = word_token_counts(tok, words_list)
        flat = [(c, tags_list[i][j] != 0, has_apos(words_list[i][j])) for i, cs in enumerate(counts) for j, c in enumerate(cs)]
        df = pd.DataFrame(flat, columns=["n_tok", "is_entity", "has_apos"])
        for name, mask in [("entity so'zlar", df.is_entity), ("entity bo'lmagan", ~df.is_entity),
                           ("apostrofli so'zlar", df.has_apos), ("apostrofsiz so'zlar", ~df.has_apos)]:
            s = df[mask]
            diag_rows.append(dict(model=mk, group=name, n_words=len(s), tokens_per_word=s.n_tok.mean(), split_rate=(s.n_tok > 1).mean()))
    PR = pd.DataFrame(probe_rows); PR.to_csv(f"{OUT}/B_tokenizer_apostrophe_probes.csv", index=False)
    DG = pd.DataFrame(diag_rows); DG.to_csv(f"{OUT}/B_tokenizer_entity_diagnostics.csv", index=False)
    print("\nApostrof variantlarining token soni (model x so'z):")
    print(PR.pivot(index="word", columns="model", values="n_tokens").to_string())
    print("\nEntity va apostrofli so'zlar bo'yicha tokenizatsiya:")
    print(DG.pivot(index="group", columns="model", values="tokens_per_word").round(3).to_string())

# %%
# ---------------- B2: NER ablation o'qitish ----------------
import dataclasses
_VALID_TA = {f.name for f in dataclasses.fields(TrainingArguments)}

def common_args(lr, seed, outdir, n_train):
    total_steps = math.ceil(n_train / NER_CFG["bs"]) * NER_CFG["epochs"]
    kw = dict(output_dir=outdir, per_device_train_batch_size=NER_CFG["bs"], per_device_eval_batch_size=64, learning_rate=lr,
              num_train_epochs=NER_CFG["epochs"], weight_decay=0.01, warmup_steps=int(0.1 * total_steps),
              eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
              metric_for_best_model="f1", greater_is_better=True, fp16=torch.cuda.is_available(), seed=seed, data_seed=seed,
              report_to="none", logging_steps=100, dataloader_num_workers=2, disable_tqdm=True)
    dropped = [k for k in kw if k not in _VALID_TA]
    if dropped: print("OGOHLANTIRISH: TrainingArguments tashlab yuborilgan parametrlar:", dropped)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in _VALID_TA})

def make_ner_metrics(names):
    def f(p):
        pred = p.predictions.argmax(-1); yt, yp = [], []
        for pr, lb in zip(pred, p.label_ids):
            yt.append([names[l] for l in lb if l != -100]); yp.append([names[x] for x, l in zip(pr, lb) if l != -100])
        pr_, rc_, f1_ = entity_prf(yt, yp)
        return {"f1": f1_, "precision": pr_, "recall": rc_}
    return f

def apply_cond(ds, cond):
    if cond == "orig":
        return ds
    target = NORM_TARGETS[cond]
    return ds.map(lambda b: {"tokens": [[norm_apos(t, target) for t in toks] for toks in b["tokens"]]}, batched=True)

def run_ner(cond, mkey, seed, lr):
    mname = MODELS[mkey]; set_seed(seed); tok = load_tok(mname); labels = NER["labels"]; outdir = f"{OUT}/tmp_run"
    def tk(batch):
        e = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=NER_CFG["max_len"])
        out = []
        for i, tags in enumerate(batch["ner_tags"]):
            prev, lab = None, []
            for w in e.word_ids(batch_index=i):
                lab.append(-100 if (w is None or w == prev) else tags[w]); prev = w
            out.append(lab)
        r = dict(e); r["labels"] = out; return r
    tr, va, te = [apply_cond(NER[s], cond).map(tk, batched=True, remove_columns=NER[s].column_names) for s in ("train", "val", "test")]
    model = AutoModelForTokenClassification.from_pretrained(mname, num_labels=len(labels), id2label=dict(enumerate(labels)),
                                                            label2id={l: i for i, l in enumerate(labels)})
    n_params = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    trainer = Trainer(model=model, args=common_args(lr, seed, outdir, len(tr)), train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorForTokenClassification(tok), processing_class=tok, compute_metrics=make_ner_metrics(labels))
    t0 = time.time(); trainer.train(); train_s = time.time() - t0
    bv = trainer.evaluate(va); out = trainer.predict(te)
    gold, pred = [], []
    for pr, lb in zip(out.predictions.argmax(-1), out.label_ids):
        gold.append([int(l) for l in lb if l != -100]); pred.append([int(x) for x, l in zip(pr, lb) if l != -100])
    os.makedirs(PRED_DIR_B, exist_ok=True)
    json.dump(dict(cond=cond, model=mkey, seed=seed, lr=lr, gold=gold, pred=pred), open(f"{PRED_DIR_B}/{cond}_{mkey}_{seed}_{lr:g}.json", "w"))
    rec = dict(cond=cond, model=mkey, seed=seed, lr=lr, params=n_params, train_time_s=round(train_s, 1),
               peak_mem_mb=int(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
               val_f1=float(bv["eval_f1"]), test_f1=float(out.metrics["test_f1"]),
               test_precision=float(out.metrics["test_precision"]), test_recall=float(out.metrics["test_recall"]))
    del trainer, model; shutil.rmtree(outdir, ignore_errors=True); gc.collect(); torch.cuda.empty_cache()
    return rec

if RUN_PART_B:
    RES_B = f"{OUT}/ner_ablation_results{'_smoke' if SMOKE else ''}.jsonl"; print("Natijalar fayli:", RES_B)
    done = set()
    if os.path.exists(RES_B):
        for line in open(RES_B):
            r = json.loads(line); done.add((r["cond"], r["model"], r["seed"], r["lr"]))
    for lr in LRS:
        for cond in CONDITIONS:
            for mkey in MODELS:
                for seed in SEEDS:
                    if (cond, mkey, seed, lr) in done: continue
                    print(f"=== {cond} | {mkey} | seed={seed} | lr={lr:g}", flush=True)
                    try:
                        rec = run_ner(cond, mkey, seed, lr)
                        open(RES_B, "a").write(json.dumps(rec, default=float) + "\n"); print({k: round(rec[k], 4) for k in ("val_f1", "test_f1")})
                    except Exception as e:
                        open(f"{OUT}/errors_B.log", "a").write(f"{cond} {mkey} {seed} {lr}: {repr(e)}\n"); print("XATO:", repr(e)[:300])
                        shutil.rmtree(f"{OUT}/tmp_run", ignore_errors=True); gc.collect(); torch.cuda.empty_cache()

# %%
# ---------------- B3: yig'ish, statistika, xato turlari ----------------
if RUN_PART_B:
    assert os.path.exists(RES_B) and os.path.getsize(RES_B) > 0, "natijalar fayli bo'sh: avval o'qitish cell'ini xatosiz bajaring (errors_B.log)"
    import io
    pd.set_option("display.float_format", "{:.4g}".format)
    RB = pd.read_json(io.StringIO(open(RES_B).read()), lines=True)
    g = RB.groupby(["lr", "cond", "model"])
    SB = g.agg(n=("test_f1", "size"), val_f1=("val_f1", "mean"), test_mean=("test_f1", "mean"), test_std=("test_f1", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan)).reset_index()
    SB.to_csv(f"{OUT}/B_summary.csv", index=False); print(SB.to_string(index=False))

    # juft taqqoslash: norm_* va orig (bir xil model, bir xil seed -> paired t-test), Holm tuzatishi
    def holm(p):
        p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
        for rank, i in enumerate(o):
            run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
        return adj
    rows = []
    for lr in RB.lr.unique():
        for cond in [c for c in RB.cond.unique() if c != "orig"]:
            for m in RB.model.unique():
                a = RB[(RB.lr == lr) & (RB.cond == "orig") & (RB.model == m)].set_index("seed").test_f1
                b = RB[(RB.lr == lr) & (RB.cond == cond) & (RB.model == m)].set_index("seed").test_f1
                common = a.index.intersection(b.index)
                if len(common) > 1:
                    d = b[common] - a[common]; t, p = stats.ttest_rel(b[common], a[common])
                    rows.append(dict(lr=lr, cond=cond, model=m, n_pairs=len(common), mean_diff_F1=d.mean(), p=p))
    if rows:
        PT = pd.DataFrame(rows); PT["p_holm"] = holm(PT.p.values); PT.to_csv(f"{OUT}/B_paired_tests_vs_orig.csv", index=False)
        print("\nPaired t-test (shart - orig), Holm tuzatishi bilan:\n", PT.to_string(index=False))

    # xato turlari va entity turi bo'yicha F1
    tax_rows, type_rows = [], []
    for f in glob.glob(f"{PRED_DIR_B}/*.json"):
        d = json.load(open(f)); cond, mkey, seed, lr = d["cond"], d["model"], d["seed"], d["lr"]
        gold = [[NAMES[i] for i in s] for s in d["gold"]]; pred = [[NAMES[i] for i in s] for s in d["pred"]]
        c = error_taxonomy(gold, pred); ng = sum(c[k] for k in ("correct", "missed", "type_only", "boundary_only", "boundary_and_type"))
        tax_rows.append(dict(cond=cond, model=mkey, seed=int(seed), lr=float(lr), n_gold=ng, **{k: c[k] / ng for k in ("correct", "missed", "type_only", "boundary_only", "boundary_and_type")}, spurious_per_gold=c["spurious"] / ng))
        for ty, v in per_type_prf(gold, pred).items():
            type_rows.append(dict(cond=cond, model=mkey, seed=int(seed), lr=float(lr), type=ty, **v))
    if tax_rows:
        TX = pd.DataFrame(tax_rows).groupby(["lr", "cond", "model"]).mean(numeric_only=True).drop(columns="seed").reset_index()
        TX.to_csv(f"{OUT}/B_error_taxonomy.csv", index=False); print("\nXato turlari (gold entity ulushi, seed o'rtachasi):\n", TX.to_string(index=False))
        TY = pd.DataFrame(type_rows).groupby(["lr", "cond", "model", "type"])[["precision", "recall", "f1"]].mean().reset_index()
        TY.to_csv(f"{OUT}/B_per_type_f1.csv", index=False); print("\nEntity turi bo'yicha F1:\n", TY.pivot_table(index=["lr", "cond", "model"], columns="type", values="f1").to_string())

    # grafik
    fig, ax = plt.subplots(figsize=(9, 4.5)); w = 0.8 / max(1, SB[['lr', 'cond']].drop_duplicates().shape[0])
    for k, ((lr, cond), s) in enumerate(SB.groupby(["lr", "cond"])):
        s = s.set_index("model").reindex(list(MODELS))
        ax.bar(np.arange(len(MODELS)) + k * w, s.test_mean, w, yerr=s.test_std, label=f"{cond} lr={lr:g}", capsize=2)
    ax.set_xticks(np.arange(len(MODELS)) + 0.4 - w / 2); ax.set_xticklabels(list(MODELS)); ax.set_ylabel("test entity F1"); ax.legend(fontsize=7)
    ax.set_ylim(max(0, SB.test_mean.min() - 0.1), 1); plt.tight_layout(); plt.savefig(f"{OUT}/B_ablation_f1.png", dpi=130); plt.close()
    print("\nTAYYOR. Natijalar:", OUT)
