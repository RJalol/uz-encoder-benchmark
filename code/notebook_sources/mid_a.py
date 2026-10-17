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
