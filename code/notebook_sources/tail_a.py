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
