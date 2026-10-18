# Before submitting to *Computer Methods and Programs in Biomedicine*

Open items. Each one is also marked in red in `main.pdf` where it applies.

- [ ] **Code link.** `main.tex` → "Data and code availability" points to https://github.com/hrid0yyy/tem-virus-siamese-fewshot, which is still **private**. Make it public, or change the URL or the sentence. Then remove the red TODO.
- [x] **Affiliation:** "Independent Researcher, Dhaka, Bangladesh" (the work was done after graduation). Add a full postal address if the submission system asks for one.
- [x] **Corresponding e-mail:** smmahbubuddinahmed@gmail.com (personal, so it won't expire).
- [x] **Competing interests:** none, as confirmed by the author.
- [ ] **Generative-AI declaration.** Confirm or adapt the statement, then remove the red "[please confirm]".
- [ ] **Final read** of the whole paper by you.
- [ ] Rebuild with `python make_numbers_tex.py --mode fp32`, `tectonic main.tex` and `tectonic cover_letter.tex`, then run `python verify_paper.py`. It must print ALL CHECKS PASSED.
- [ ] **Upload to Editorial Manager:**
  - `main.tex` + `numbers.tex` + `references.bib` + `elsarticle-num.bst` + `figures/architecture.pdf`, or the PDF if requested;
  - `highlights.txt`, as a separate file with "highlights" in its name;
  - `cover_letter.pdf`;
  - Fig. 1 as a separate file (`figures/architecture.pdf`).
