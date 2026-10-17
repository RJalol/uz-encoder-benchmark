import os,io,glob,json,math,re,unicodedata,time,dataclasses,shutil,gc
import numpy as np,pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/tmp/nb/out_b"; shutil.rmtree(OUT,ignore_errors=True); os.makedirs(OUT); SMOKE=True
exec(open('/tmp/nb/pure.py').read())
src=open('/tmp/nb/tail_b.py').read()
K=3; NAMES=list("abc"); TRAIN_CNT=np.array([100,50,20]); V1_RES=[]
MODELS={"m1":"x","m2":"y"}; CONFIGS={"e_lr3":dict(lr=3e-5,epochs=6,cond="orig"),"e_lr5":dict(lr=5e-5,epochs=6,cond="orig")}
BASE=dict(bs=16,ga=2,max_len=512); RESUME_MIN=0; PILOT_SEEDS=[13]; FINAL_SEEDS=[13,21]; TIME_BUDGET_H=11
TE=pd.DataFrame({"text":["a b"]*5}); 
def load_tok(n):
    class T:
        def __call__(s,x,**k): return {"input_ids":[[1,2,3]]*len(x)}
    return T()
rng=np.random.default_rng(1)
def run_cfg(c,m,s):
    cfg={**BASE,**CONFIGS[c]}
    return dict(cfg=c,model=m,seed=s,lr=cfg["lr"],epochs=6,cond=cfg["cond"],max_len=512,test_macro_f1=float(rng.random()),test_accuracy=.5,val_macro_f1=float(rng.random()),
                best_epoch=5.0,n_never_pred=0,per_class_f1=[.1,.2,.3],source="trained",test_weighted_f1=.5)
# skip truncation cell (needs tok): run from MAIN cell on
cells=src.split("# %%\n")
for c in cells:
    if "TRUNCATION" in c: continue
    exec(c)
print(open(OUT+"/news_v2_summary.csv").read())
# resume test: rerun main+agg with the file present
RECS_before=len(RECS)
exec([c for c in cells if "MAIN:" in c][0]); print("resume ok", RECS_before, len(RECS))
