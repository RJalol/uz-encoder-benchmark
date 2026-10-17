import os,io,glob,json,math,re,unicodedata,time,dataclasses
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,precision_recall_fscore_support
from collections import Counter
OUT="/tmp/nb/out_a"; os.makedirs(OUT,exist_ok=True); SMOKE=True
exec(open('/tmp/nb/pure.py').read())
rng=np.random.default_rng(0)
K=3; NAMES=["a","b","c"]
def mk(n):
    lab=rng.integers(0,K,n); ap=["'","ʻ","‘"]
    return pd.DataFrame({"text":[f"o{ap[l]}zbek gap {rng.integers(0,50)} so{ap[l]}z"+" kalit"+str(l)*(rng.integers(0,2)) for l in lab],"label":lab})
TR,VA,TE=mk(400),mk(100),mk(100)
src=open('/tmp/nb/mid_a.py').read()
src=src.split("# %%\n# ---------------- 3)")[0]
exec(src)
