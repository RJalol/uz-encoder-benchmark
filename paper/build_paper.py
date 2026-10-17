import json, numpy as np, pandas as pd
from scipy import stats

MAIN="/tmp/res_final"; ANA="/tmp/res3/analysis"; V2="/tmp/res9/news_fairness_v2"; DG="/tmp/res6/news_diag"
f3=lambda x:f"{x:.3f}"
pm=lambda m,s:f"${m:.3f}\\pm{s:.3f}$"
def pfmt(p):
    return "<0.001" if p<0.001 else f"{p:.3f}"
DN={"tahrirchi-base":"TahrirchiBERT-base","tahrirchi-small":"TahrirchiBERT-small","xlmr-base":"XLM-R-base","mbert":"mBERT"}
ORDER=["xlmr-base","mbert","tahrirchi-base","tahrirchi-small"]

# ---------- load ----------
S=pd.read_csv(f"{MAIN}/summary.csv").set_index(["dataset","model"])
ST=pd.read_csv(f"{MAIN}/stat_tests.csv")
TK=pd.read_csv(f"{MAIN}/tokenizer_stats.csv").set_index(["dataset","model"])
R0=pd.DataFrame([json.loads(l) for l in open(f"{MAIN}/results.jsonl")])
params=R0.groupby("model").params.first()  # sib200 head
V2S=pd.read_csv(f"{V2}/news_v2_summary.csv"); V2T=pd.read_csv(f"{V2}/news_v2_tests.csv"); V2N=pd.read_csv(f"{V2}/news_v2_orig_vs_ascii.csv")
V2R=pd.DataFrame([json.loads(l) for l in open(f"{V2}/news_v2_results.jsonl")])
TR512=pd.read_csv(f"{V2}/truncation_diagnostics_512.csv").set_index("model")
V1V2=pd.read_csv(f"{V2}/v1_vs_v2.csv").set_index("model")
names=pd.read_csv(f"{V2}/news_v2_per_class_f1.csv")["class"].tolist(); ntrain=pd.read_csv(f"{V2}/news_v2_per_class_f1.csv")["n_train"].tolist()
CEN=pd.read_csv(f"{DG}/apostrophe_census_news.csv"); BYC=pd.read_csv(f"{DG}/apostrophe_by_class.csv"); SC=pd.read_csv(f"{DG}/shortcut_check.csv",index_col=0)
NS=pd.read_csv(f"{ANA}/B_summary.csv"); NP=pd.read_csv(f"{ANA}/B_paired_tests_vs_orig.csv"); NT=pd.read_csv(f"{ANA}/B_error_taxonomy.csv")

def st(ds,a,b):
    r=ST[(ST.dataset==ds)&(((ST.A==a)&(ST.B==b)))]
    return r.iloc[0]
def v2t(setting,a,b):
    r=V2T[(V2T.setting==setting)&(V2T.A==a)&(V2T.B==b)]
    return r.iloc[0]

# selected cfgs
sel={"mbert":"e_lr3","tahrirchi-base":"e_lr3","tahrirchi-small":"e_lr5","xlmr-base":"e_lr3"}
lrs={"e_lr3":"3\\times10^{-5}","e_lr5":"5\\times10^{-5}"}
J=names.index("Jinoyat")
pc={}
for m,c in sel.items():
    for lab,cfg in [("orig",c),("ascii",c+"__ascii")]:
        g=V2R[(V2R.model==m)&(V2R.cfg==cfg)].sort_values("seed"); pc[(m,lab)]=np.array(g.per_class_f1.tolist())
def v2row(setting,m): return V2S[(V2S.setting==setting)&(V2S.model==m)].iloc[0]
def excl(m,lab): return np.delete(pc[(m,lab)],J,axis=1).mean(1)
jin=lambda m,lab:(pc[(m,lab)][:,J].mean(),pc[(m,lab)][:,J].std(ddof=1))

# ---------- tables ----------
T1=[]
for m in ORDER:
    tok={"xlmr-base":"SentencePiece (multilingual)","mbert":"WordPiece, cased (multilingual)","tahrirchi-base":"Byte-level BPE, vocab. 30,528 (Uzbek)","tahrirchi-small":"Byte-level BPE, vocab. 30,528 (Uzbek)"}[m]
    T1.append(f"{DN[m]} & {params[m]/1e6:.0f} & {tok} \\\\")
T1="\n".join(T1)

T2=[]
for m in ORDER:
    cells=[pm(S.loc[(d,m),'mean'],S.loc[(d,m),'std']) for d in ["sib200","uz_news","wikiann_uz"]]
    T2.append(f"{DN[m]} & "+" & ".join(cells)+" \\\\")
T2="\n".join(T2)

T3=[]
for m in ORDER:
    c=[]
    for d in ["sib200","uz_news","wikiann_uz"]:
        r=TK.loc[(d,m)]; c.append(f"{r.fertility:.2f} / {100*r.split_word_rate:.0f} / {100*r.unk_rate:.2f}")
    T3.append(f"{DN[m]} & "+" & ".join(c)+" \\\\")
T3="\n".join(T3)

T4=[]
for m in ORDER:
    o,a=v2row("orig",m),v2row("norm_ascii",m); d=V2N[V2N.model==m].iloc[0]
    T4.append(f"{DN[m]} & ${lrs[sel[m]]}$ & {pm(o.test_macro_f1,o['std'])} & {pm(a.test_macro_f1,a['std'])} & ${d.mean_diff_ascii_minus_orig:+.3f}$ ({pfmt(d.p_holm)}) \\\\")
T4="\n".join(T4)

T5=[]
for m in ORDER:
    jo,js=jin(m,"orig"); ja,jsa=jin(m,"ascii")
    T5.append(f"{DN[m]} & {pm(jo,js)} & {pm(ja,jsa)} & {excl(m,'orig').mean():.3f} & {excl(m,'ascii').mean():.3f} \\\\")
T5="\n".join(T5)

def frac(cls,col): return BYC[BYC['class']==cls][col].iloc[0]
rowsT6=[]
for cls,disp in [("Jinoyat","Jinoyat"),("Qonunchilik","Qonunchilik"),("Siyosat","Siyosat"),("Salomatlik","Salomatlik"),("Sport","Sport"),("O‘zbekiston","O\\textquoteleft zbekiston")]:
    n=int(BYC[BYC['class']==cls].n_docs.iloc[0])
    rowsT6.append(f"{disp} & {n} & {100*frac(cls,'frac_ASCII'):.1f} & {100*frac(cls,'frac_U+2018'):.1f} & {100*frac(cls,'frac_U+2019'):.1f} & {100*frac(cls,'frac_U+0060'):.1f} & {100*frac(cls,'frac_U+02BB'):.1f} \\\\")
T6="\n".join(rowsT6)

T7=[]
for lab,k in [("Majority class","majority_class"),("Apostrophe-character counts only (logistic regression)","apostrophe_only_LR"),
              ("Word TF-IDF, original text","tfidf_word_orig"),("Word TF-IDF, ASCII-normalized","tfidf_word_norm_ascii"),
              ("Character 1--4-gram TF-IDF, original text","tfidf_char_orig"),("Character 1--4-gram TF-IDF, ASCII-normalized","tfidf_char_norm_ascii")]:
    T7.append(f"{lab} & {SC.loc[k,'test_macro_f1']:.3f} \\\\")
T7="\n".join(T7)

T8=[]
for i,(n,c) in enumerate(zip(names,ntrain)):
    disp=n.replace("‘","\\textquoteleft ")
    T8.append(f"{disp} & {c} \\\\")
# 2-column class-count table (15 rows -> 8 + 7)
half=8
T8rows=[]
left=[(names[i].replace("‘","\\textquoteleft "),ntrain[i]) for i in range(half)]
right=[(names[i].replace("‘","\\textquoteleft "),ntrain[i]) for i in range(half,len(names))]
for i in range(half):
    r=right[i] if i<len(right) else ("","")
    T8rows.append(f"{left[i][0]} & {left[i][1]} & {r[0]} & {r[1]} \\\\")
T8="\n".join(T8rows)

# NER ablation numbers
def ner(m,cond): return NS[(NS.model==m)&(NS.cond==cond)].iloc[0]
ner_rows=[]
for m in ORDER:
    o=ner(m,"orig"); a=ner(m,"norm_ascii"); p=NP[NP.model==m].iloc[0]
    ner_rows.append(f"{DN[m]} & {pm(o.test_mean,o.test_std)} & {pm(a.test_mean,a.test_std)} & ${p.mean_diff_F1:+.3f}$ ({pfmt(p.p_holm)}) \\\\")
T9="\n".join(ner_rows)
tax=NT[NT.cond=="orig"].set_index("model")

# scalar facts for text
X=lambda d,m:S.loc[(d,m),'mean']
newsbase_collapse={m:(int((R0[(R0.dataset=='uz_news')&(R0.model==m)].shape[0])), ) for m in ORDER}
tb_x=v2t("orig","tahrirchi-base","xlmr-base"); tb_x_a=v2t("norm_ascii","tahrirchi-base","xlmr-base")
tb_m=v2t("orig","mbert","tahrirchi-base"); m_x_a=v2t("norm_ascii","mbert","xlmr-base")
ts_x=v2t("orig","tahrirchi-small","xlmr-base")
tb_ts=v2t("orig","tahrirchi-base","tahrirchi-small")
ex_o={m:excl(m,"orig").mean() for m in ORDER}; ex_a={m:excl(m,"ascii").mean() for m in ORDER}
tbj=jin("tahrirchi-base","orig"); tbja=jin("tahrirchi-base","ascii"); xj=jin("xlmr-base","orig"); xja=jin("xlmr-base","ascii"); mj=jin("mbert","orig"); mja=jin("mbert","ascii"); tsj=jin("tahrirchi-small","orig"); tsja=jin("tahrirchi-small","ascii")
census=CEN.set_index("codepoint")
def cfrac(cp): return 100*census.loc[cp,"frac_docs"]
g=lambda ds,a,b:st(ds,a,b)
sib_tb_ts=g("sib200","tahrirchi-base","tahrirchi-small"); sib_m_tb=g("sib200","mbert","tahrirchi-base")
wk_m_x=g("wikiann_uz","mbert","xlmr-base"); wk_tb_x=g("wikiann_uz","tahrirchi-base","xlmr-base"); wk_m_tb=g("wikiann_uz","mbert","tahrirchi-base")
nw_tb_x=g("uz_news","tahrirchi-base","xlmr-base"); nw_m_x=g("uz_news","mbert","xlmr-base"); nw_m_tb=g("uz_news","mbert","tahrirchi-base")
nbase=R0[R0.dataset=="uz_news"]
# collapse counts from baseline preds: from v1 summary (seeds with collapse) -> read preds
import glob, os
Yt=np.load("/tmp/res6/news_diag/test_labels.npy")
col={}
for m in ORDER:
    c=[]
    for s in [13,21,42,87,100]:
        p=np.load(f"{MAIN}/preds/uz_news_{m}_{s}.npy"); c.append(int((np.bincount(p,minlength=15)==0).sum()))
    col[m]=c
drops=V2N.set_index("model").mean_diff_ascii_minus_orig
dmin,dmax=drops.min(),drops.max()
excl_drop={m:ex_a[m]-ex_o[m] for m in ORDER}
nerp=NP.set_index("model")
ner_maxabs=NP.mean_diff_F1.abs().max()
print("collapse",col); print("excl_o",ex_o); print("excl_a",ex_a)

tex=open("/tmp/paper/template_body.tex").read()
vals=dict(locals())
import re
def peq(p): return "p<0.001" if p<0.001 else f"p={p:.3f}"
vals["peq"]=peq
def sub(m):
    e=m.group(1).strip(); fm=re.search(r":(\.\d+f)$",e)
    if fm:
        return format(eval(e[:fm.start()],globals(),vals),fm.group(1))
    return str(eval(e,globals(),vals))
tex=re.sub(r"@@(.+?)@@",sub,tex,flags=re.S)
open("/tmp/paper/paper.tex","w").write(tex)
print("written", len(tex))
