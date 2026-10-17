"""Approximate word counts of main.tex against the CMPB limits (abstract <= 250-350, main text <= 3,500)."""
import re
from pathlib import Path

s = Path(__file__).with_name("main.tex").read_text(encoding="utf-8")


def words(t):
    t = re.sub(r"\\begin\{table\}.*?\\end\{table\}", "", t, flags=re.S)
    t = re.sub(r"(?<!\\)%.*", "", t)
    t = re.sub(r"\\(cite|ref|label|cmidrule)\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " X ", t)  # each macro counts as one word (numbers)
    t = re.sub(r"[{}$\\~]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


abstract = s[s.index(r"\begin{abstract}"):s.index(r"\end{abstract}")]
main = s[s.index(r"\section{Introduction}"):s.index(r"\section*{Declaration of competing")]
print(f"abstract  ~{words(abstract)} words (limit 250 general / 350 for original research)")
print(f"main text ~{words(main)} words (limit 3,500, excluding abstract, tables, references, declarations)")
