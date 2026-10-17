import json, sys, re
def cells(src):
    parts = re.split(r"(?m)^# %%( \[markdown\])?\n", src)
    out = []; i = 1
    # parts: [pre, flag1, body1, flag2, body2...]
    while i < len(parts):
        flag, body = parts[i], parts[i + 1]; i += 2
        if flag:
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.rstrip("\n").split("\n"))
            out.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.rstrip("\n").splitlines(True)})
    return out
def build(files, py_out, nb_out):
    src = "\n".join(open(f).read().rstrip("\n") + "\n" for f in files)
    open(py_out, "w").write(src)
    nb = {"cells": cells(src), "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(nb_out, "w"), ensure_ascii=False, indent=1)
    compile(src, py_out, "exec")
    print(nb_out, len(nb["cells"]), "cells")
if __name__ == "__main__":
    kind = sys.argv[1]
    if kind == "a":
        build(["head_a.py", "pure.py", "data.py", "common.py", "mid_a.py", "tail_a.py"], "/tmp/uz_news_norm_diag.py", "/tmp/uz_news_norm_diag.ipynb")
    else:
        build(["head_b.py", "pure.py", "data.py", "common.py", "tail_b.py"], "/tmp/uz_news_fairness_v2.py", "/tmp/uz_news_fairness_v2.ipynb")
