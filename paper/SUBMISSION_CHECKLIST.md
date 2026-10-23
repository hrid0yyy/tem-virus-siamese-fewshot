# Before submitting to *Computer Methods and Programs in Biomedicine*

Open items. Each one is also marked in red in `main.pdf` where it applies.

- [x] **Code link.** `main.tex` → "Data and code availability" points to https://github.com/hrid0yyy/tem-virus-siamese-fewshot. Red TODO removed; privacy audit passed (2026-10-23); repository made public by Claude.
- [x] **Affiliation:** "Independent Researcher, Dhaka, Bangladesh" (the work was done after graduation). Add a full postal address if the submission system asks for one.
- [x] **Corresponding e-mail:** smmahbubuddinahmed@gmail.com (personal, so it won't expire).
- [x] **Competing interests:** none, as confirmed by the author.
- [x] **Generative-AI declaration:** uses Elsevier's template wording, plus a matching sentence in Methods 2.7 (Implementation), as approved by the author.
- [ ] **Final read** of the whole paper by you.
- [ ] Rebuild with `python make_numbers_tex.py --mode fp32`, `tectonic main.tex` and `tectonic cover_letter.tex`, then run `python verify_paper.py`. It must print ALL CHECKS PASSED.
- [ ] **Upload to Editorial Manager:**
  - `main.tex` + `numbers.tex` + `references.bib` + `elsarticle-num.bst` + `figures/architecture.pdf`, or the PDF if requested;
  - `highlights.txt`, as a separate file with "highlights" in its name;
  - `cover_letter.pdf`;
  - every figure as a separate file: `figures/dataset.pdf` (Fig. 1), `figures/architecture.pdf` (Fig. 2), `figures/confusion_rare.pdf` (Fig. 3), `figures/kshot.pdf` (Fig. 4) and `figures/training.pdf` (Fig. B.5).
