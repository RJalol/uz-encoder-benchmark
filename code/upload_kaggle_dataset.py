#!/usr/bin/env python3
"""
Kaggle Dataset'ni kompyuterdan (brauzersiz) yaratish/yangilash.

Ishlatish:
  1) pip install kaggle
  2) Kaggle API tokenini oling: kaggle.com -> Settings -> API -> "Create New Token" (kaggle.json yuklanadi).
     kaggle.json ni  ~/.kaggle/kaggle.json  (Windows: C:\\Users\\<ism>\\.kaggle\\kaggle.json)  ga qo'ying.
  3) Zipni shu skriptga bering:
       python upload_kaggle_dataset.py --user SIZNING_KAGGLE_USERNAME --zip uz-news-v2-part1.zip --slug uz-news-v2-part1
       python upload_kaggle_dataset.py --user SIZNING_KAGGLE_USERNAME --zip uz-prev-results.zip   --slug uz-prev-results
  Dataset PRIVATE yaratiladi. Allaqachon mavjud bo'lsa, yangi versiya sifatida yuklanadi.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, zipfile

def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip()); 
    if r.stderr.strip(): print(r.stderr.strip(), file=sys.stderr)
    return r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username (profil URL'idagi nom)")
    ap.add_argument("--zip", required=True, help="yuborilgan zip fayl yo'li")
    ap.add_argument("--slug", required=True, help="dataset nomi, masalan uz-news-v2-part1")
    ap.add_argument("--title", default=None, help="sarlavha (default = slug)")
    ap.add_argument("--dry-run", action="store_true", help="faqat tayyorlaydi, yuklamaydi")
    a = ap.parse_args()

    if not a.dry_run and shutil.which("kaggle") is None:
        sys.exit("`kaggle` buyrug'i topilmadi: avval  pip install kaggle  qiling va kaggle.json ni ~/.kaggle/ ga qo'ying.")
    if not os.path.isfile(a.zip): sys.exit(f"zip topilmadi: {a.zip}")

    work = tempfile.mkdtemp(prefix="kds_")
    with zipfile.ZipFile(a.zip) as z:
        for n in z.namelist():                       # xavfsizlik: yo'l chiqib ketmasin
            if os.path.isabs(n) or ".." in n.replace("\\", "/").split("/"): sys.exit(f"xavfli yo'l zipda: {n}")
        z.extractall(work)
    # zip ichida bitta yuqori papka bo'lsa (uz-news-v2-part1/...), shuni ildiz qilamiz
    items = [x for x in os.listdir(work) if not x.startswith("__MACOSX")]
    root = os.path.join(work, items[0]) if len(items) == 1 and os.path.isdir(os.path.join(work, items[0])) else work

    meta = {"title": a.title or a.slug, "id": f"{a.user}/{a.slug}", "licenses": [{"name": "unknown"}]}
    json.dump(meta, open(os.path.join(root, "dataset-metadata.json"), "w"), indent=2)
    files = [os.path.relpath(os.path.join(d, f), root) for d, _, fs in os.walk(root) for f in fs]
    print("Yuklanadigan fayllar:", len(files)); [print("  ", f) for f in sorted(files)[:15]]
    if a.dry_run: print("dry-run: tayyor papka:", root); return

    r = run(["kaggle", "datasets", "create", "-p", root, "--dir-mode", "zip"])   # private (default)
    out = (r.stdout + r.stderr).lower()
    if r.returncode != 0 or "already" in out or "error" in out:
        print("Mavjud bo'lishi mumkin -> yangi versiya yuklanadi...")
        r = run(["kaggle", "datasets", "version", "-p", root, "-m", "update", "--dir-mode", "zip"])
    if r.returncode == 0:
        print(f"\nTAYYOR: https://www.kaggle.com/datasets/{a.user}/{a.slug}")
    else:
        sys.exit("Yuklash xato bilan tugadi, yuqoridagi xabarni yuboring.")

if __name__ == "__main__":
    main()
