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
