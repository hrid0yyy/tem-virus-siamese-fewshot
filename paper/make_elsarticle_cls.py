"""Extract elsarticle.cls (v3.4, Elsevier's official bundle in ../elsarticle/) from elsarticle.dtx, doing what
docstrip does for the single <class> block: drop '%' comment lines, keep '%%' lines and code.
(tectonic generates the file from elsarticle.ins in memory only, so it cannot be used for this.)

    python make_elsarticle_cls.py   -> elsarticle.cls next to main.tex (overrides tectonic's bundled v3.3)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
dtx = (HERE.parent / "elsarticle" / "elsarticle.dtx").read_text(encoding="utf-8").splitlines()
a, b = dtx.index("%<*class>"), dtx.index("%</class>")
body = [ln for ln in dtx[a + 1:b] if ln.startswith("%%") or not ln.startswith("%")]
assert not any(ln.startswith("%<") for ln in body), "nested docstrip guards are not handled"
head = ["%% elsarticle.cls, extracted from the official elsarticle.dtx (Elsarticle Bundle, Elsevier Ltd, LPPL 1.3+)",
        "%% by make_elsarticle_cls.py; equivalent to running docstrip on elsarticle.ins."]
(HERE / "elsarticle.cls").write_text("\n".join(head + body) + "\n", encoding="utf-8", newline="\n")
print("wrote elsarticle.cls", len(body), "lines")
