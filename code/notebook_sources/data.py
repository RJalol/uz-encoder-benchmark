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
