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
