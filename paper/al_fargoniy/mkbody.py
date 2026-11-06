import re
s=open('/tmp/paper/paper.tex',encoding='utf8').read()
b=s[s.index(r"\label{firstpage}")+len(r"\label{firstpage}"):s.index(r"\begin{center}\textbf{REFERENCES}")]
# --- citation renumbering (old alphabetical -> order of first appearance)
mp={3:1,2:2,7:3,6:4,8:5,9:6,11:7,4:8,5:9,1:10,10:11}
def fix(m): return "["+", ".join(map(str,sorted(mp[int(x)] for x in m.group(1).split(','))))+"]"
b=re.sub(r"(?<![\w\)\]])\[(\d+(?:,\d+)*)\]",fix,b)
b=re.sub(r"\[(\d+), (\d+)\]",r"[\1], [\2]",b)

def mathfix(m):
    t=m.group(1)
    sup={'-':'⁻','4':'⁴','5':'⁵'}
    t=re.sub(r"\^\{(-?\d)\}",lambda x:"\\textsuperscript{"+x.group(1).replace("-","−")+"}",t)
    t=t.replace(r"\times"," × ").replace(r"\pm"," ± ").replace(r"\approx","≈ ")
    t=re.sub(r"^-","−",t)
    t=re.sub(r"\b([ptn])\b(?=\s*[=<>]|$)",r"\\emph{\1}",t)
    t=re.sub(r"([=<>])",r" \1 ",t).replace("  "," ")
    return t.strip() if t.strip().startswith("\\emph") or True else t
b=re.sub(r"\$([^$]*)\$",mathfix,b)

# --- headings
b=b.replace(r"\textbf{Data, models and experimental protocol}","\\textbf{Literature review and methodology}")
b=re.sub(r"\\begin\{center\}\s*\\textbf\{Discussion and limitations\}\s*\\end\{center\}",r"\\subsection*{Discussion and limitations}",b)
b=re.sub(r"\\begin\{center\}\s*\\textbf\{([^}]*)\}\s*\\end\{center\}",r"\\section*{\1}",b)
# --- tables: caption paragraph then tabular
b=b.replace("\\\\[2pt]\n\\begin{tabular}","\n\n\\begin{tabular}")
b=b.replace("\\begin{center}","").replace("\\end{center}","")
# --- enumerate -> manual numbering
def enum(m):
    items=[x.strip() for x in re.split(r"\\item\s+",m.group(1)) if x.strip()]
    return "\n\n".join(f"({i+1}) {t}" for i,t in enumerate(items))
b=re.sub(r"\\begin\{enumerate\}(.*?)\\end\{enumerate\}",enum,b,flags=re.S)
b=re.sub(r"\\url\{([^}]*)\}",r"\\texttt{\1}",b)
b=b.replace("\\textquoteleft ","‘").replace(r"\textquoteleft","‘").replace(r"\textquoteright","’").replace("favour","favor")
# --- related work paragraph
rw=("\\textbf{Related work.} Multilingual encoders such as mBERT [1] and XLM-R [2] are pretrained on text in many languages, whereas "
"TahrirchiBERT [3] was pretrained on Uzbek text only. Rust et al. [7] examined how tokenizer quality affects the performance of multilingual models, "
"which motivates our tokenizer statistics. Annotation artifacts and shortcut learning have been documented for other tasks [8], [9], which motivates "
"our check of typographic cues. For Uzbek, the news corpus of Kuriyozov et al. [4] provides the classification data, and sentiment resources have been "
"described in [5] and [6]; SIB-200 [10] and WikiANN [11] provide topic classification and name-tagging data.\n\n")
b=b.replace("\\textbf{Datasets.}",rw+"\\textbf{Datasets.}",1)
# --- declarations: add required statements
decl_add=("\\textbf{Funding.} This research received no external funding.\n\n"
"\\textbf{Conflict of interest.} The author declares no conflict of interest.\n\n"
"\\textbf{Use of generative AI tools.} The generative AI assistant Claude (Anthropic) was used to help write the analysis code, to draft and edit the English text and to prepare this manuscript format. The author designed the study, checked all results and references, and is responsible for the scientific content, the reliability of the results and compliance with copyright requirements.\n\n")
i=b.index("\\section*{Declarations}")+len("\\section*{Declarations}")
b=b[:i]+"\n\n"+decl_add+b[i:].lstrip()
refs=r"""
\section*{References}

[1] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, ``BERT: Pre-training of deep bidirectional transformers for language understanding,'' in \emph{Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol., Vol. 1}, Minneapolis, MN, USA, 2019, pp. 4171--4186, doi: 10.18653/v1/N19-1423.

[2] A. Conneau \emph{et al.}, ``Unsupervised cross-lingual representation learning at scale,'' in \emph{Proc. 58th Annu. Meeting Assoc. Comput. Linguistics}, 2020, pp. 8440--8451, doi: 10.18653/v1/2020.acl-main.747.

[3] M. Mamasaidov and A. Shopulatov, ``TahrirchiBERT base,'' Hugging Face model card, 2023. [Online]. Available: https://huggingface.co/tahrirchi/tahrirchi-bert-base

[4] E. Kuriyozov, U. Salaev, S. Matlatipov, and G. Matlatipov, ``Text classification dataset and analysis for Uzbek language,'' \emph{arXiv preprint} arXiv:2302.14494, 2023.

[5] S. Matlatipov, H. Rahimboeva, J. Rajabov, and E. Kuriyozov, ``Uzbek sentiment analysis based on local restaurant reviews,'' \emph{arXiv preprint} arXiv:2205.15930, 2022.

[6] S. G. Matlatipov, J. Rajabov, E. Kuriyozov, and M. Aripov, ``UzABSA: Aspect-based sentiment analysis for the Uzbek language,'' in \emph{Proc. 3rd Annu. Meeting Special Interest Group Under-Resourced Languages @ LREC-COLING 2024}, Torino, Italy, 2024, pp. 394--403.

[7] P. Rust, J. Pfeiffer, I. Vuli\'c, S. Ruder, and I. Gurevych, ``How good is your tokenizer? On the monolingual performance of multilingual language models,'' in \emph{Proc. 59th Annu. Meeting Assoc. Comput. Linguistics and 11th Int. Joint Conf. Natural Lang. Process. (Vol. 1: Long Papers)}, 2021, pp. 3118--3135, doi: 10.18653/v1/2021.acl-long.243.

[8] R. Geirhos \emph{et al.}, ``Shortcut learning in deep neural networks,'' \emph{Nature Mach. Intell.}, vol. 2, no. 11, pp. 665--673, 2020, doi: 10.1038/s42256-020-00257-z.

[9] S. Gururangan, S. Swayamdipta, O. Levy, R. Schwartz, S. R. Bowman, and N. A. Smith, ``Annotation artifacts in natural language inference data,'' in \emph{Proc. 2018 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol., Vol. 2}, New Orleans, LA, USA, 2018, pp. 107--112, doi: 10.18653/v1/N18-2017.

[10] D. I. Adelani \emph{et al.}, ``SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects,'' in \emph{Proc. EACL 2024 (Vol. 1: Long Papers)}, 2024, pp. 226--245, doi: 10.18653/v1/2024.eacl-long.14.

[11] X. Pan, B. Zhang, J. May, J. Nothman, K. Knight, and H. Ji, ``Cross-lingual name tagging and linking for 282 languages,'' in \emph{Proc. 55th Annu. Meeting Assoc. Comput. Linguistics (Vol. 1)}, Vancouver, Canada, 2017, pp. 1946--1958, doi: 10.18653/v1/P17-1178.
"""
open('body.tex','w',encoding='utf8').write("\\documentclass{article}\\begin{document}\n"+b.strip()+"\n"+refs+"\n\\end{document}\n")
print(len(b.split()))
