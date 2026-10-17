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
