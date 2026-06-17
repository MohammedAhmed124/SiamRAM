#!/usr/bin/env bash
# Rebuild SiamRAM_Theory_Course.pdf from the numbered Markdown chapters.
# Self-bootstrapping: creates a venv, installs a bundled pandoc (pypandoc_binary),
# downloads the self-contained tectonic LaTeX engine, then renders to PDF.
# Math is rendered by real (Xe)LaTeX; ASCII diagrams use DejaVu Sans Mono.
set -euo pipefail
cd "$(dirname "$0")"
DIR="$(pwd)"; BUILD="${TMPDIR:-/tmp}/siamram_pdfbuild"; mkdir -p "$BUILD"
VENV="$BUILD/venv"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet pypandoc_binary

# tectonic (static binary) — download once
TEC="$BUILD/tectonic"
if [ ! -x "$TEC" ]; then
  URL=$("$VENV/bin/python" - <<'PY'
import json,urllib.request
d=json.load(urllib.request.urlopen("https://api.github.com/repos/tectonic-typesetting/tectonic/releases?per_page=20"))
for r in d:
    if r.get("tag_name","").startswith("tectonic@"):
        for a in r["assets"]:
            if "x86_64-unknown-linux-musl" in a["name"] and a["name"].endswith(".tar.gz"):
                print(a["browser_download_url"]); raise SystemExit
PY
)
  "$VENV/bin/python" -c "import urllib.request,sys;urllib.request.urlretrieve('$URL','$BUILD/t.tgz')"
  tar xzf "$BUILD/t.tgz" -C "$BUILD"; chmod +x "$TEC"
fi

# 1) concatenate chapters; substitute two glyphs absent from DejaVu (PDF only)
"$VENV/bin/python" - <<PY
import glob
files=sorted(glob.glob("[01]*.md"))
t="\n\n".join(open(f,encoding="utf-8").read().rstrip()+"\n" for f in files)
t=t.replace("ℓ","l").replace("✓","\$\\\\checkmark\$")  # U+2113 ell, U+2713 check
open("$BUILD/combined.md","w",encoding="utf-8").write(t)
PY

# 2) render
PATH="$BUILD:$PATH" "$VENV/bin/python" - <<PY
import pypandoc, os
os.environ["PATH"]="$BUILD:"+os.environ["PATH"]
extra=["-H","$DIR/pdf_header.tex","--pdf-engine=tectonic",
 "-V","documentclass=report","-V","geometry:a4paper,margin=0.6in","-V","fontsize=11pt","-V","linestretch=1.05",
 "-V","mainfont=DejaVu Serif","-V","sansfont=DejaVu Sans","-V","monofont=DejaVu Sans Mono",
 "-V","colorlinks=true","-V","linkcolor=RoyalBlue","-V","toccolor=RoyalBlue","-V","urlcolor=RoyalBlue",
 "-V","title=SiamRAM, From the Ground Up","-V","subtitle=A Complete Mathematical and Theoretical Course",
 "-V","author=A theory course derived directly from the SiamRAM implementation","-V","date=2026-06-13",
 "--toc","--toc-depth=2"]
pypandoc.convert_file("$BUILD/combined.md","pdf",outputfile="$DIR/SiamRAM_Theory_Course.pdf",extra_args=extra)
print("wrote $DIR/SiamRAM_Theory_Course.pdf")
PY
