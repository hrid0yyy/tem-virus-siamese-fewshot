"""Build the flat upload folder for Micron (Editorial Manager compiles LaTeX itself and wants each figure as a
separate file).  Run after `tectonic main.tex` and `tectonic cover_letter.tex`:

    python make_submission.py      -> submission_micron/  (+ submission_micron.zip)

The folder holds main.tex with flat figure names, numbers.tex, references.bib, main.bbl, elsarticle-num.bst,
Figure_1.pdf ... Figure_B1.pdf, highlights.txt, cover_letter.pdf and main.pdf (reference copy only).
"""
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission_micron"
FIGS = {  # source figure -> upload name, in order of appearance (Fig. 1-4, Fig. B.1)
    "figures/dataset.pdf": "Figure_1.pdf",
    "figures/architecture.pdf": "Figure_2.pdf",
    "figures/confusion_rare.pdf": "Figure_3.pdf",
    "figures/kshot.pdf": "Figure_4.pdf",
    "figures/training.pdf": "Figure_B1.pdf",
}

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir()

tex = (HERE / "main.tex").read_text(encoding="utf-8")
used = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
assert sorted(used) == sorted(FIGS), f"figures in main.tex changed: {used}"
for src, dst in FIGS.items():
    tex = tex.replace("{" + src + "}", "{" + dst + "}")
    shutil.copy2(HERE / src, OUT / dst)
(OUT / "main.tex").write_text(tex, encoding="utf-8", newline="\n")
for f in ("numbers.tex", "references.bib", "main.bbl", "elsarticle-num.bst", "highlights.txt", "cover_letter.pdf",
          "main.pdf"):
    shutil.copy2(HERE / f, OUT / f)

# pdfLaTeX on the submission system may reject non-ASCII input that tectonic (XeTeX) accepts
for f in ("main.tex", "numbers.tex", "references.bib", "main.bbl"):
    bad = [(i + 1, ln) for i, ln in enumerate((OUT / f).read_text(encoding="utf-8").splitlines())
           if any(ord(ch) > 127 for ch in ln)]
    assert not bad, f"non-ASCII characters in {f}: {bad[:3]}"

with zipfile.ZipFile(HERE / "submission_micron.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.iterdir()):
        z.write(f, f.name)
print(f"wrote {OUT} ({len(list(OUT.iterdir()))} files) and submission_micron.zip")
