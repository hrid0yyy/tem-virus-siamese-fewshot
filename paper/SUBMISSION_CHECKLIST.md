# Before submitting to *Micron* (Elsevier)

History: submitted to *Computer Methods and Programs in Biomedicine* (CMPB-D-26-08813), desk-rejected in November 2026 for "lack of sufficient novelty". The submitted version is tagged `cmpb-submission` in git. Rules below are from Micron's Guide for Authors, read on 19 Nov 2026.

## Done in the manuscript
- [x] **Title** reframed for microscopists: "Identifying rare viruses in negative-stain transmission electron microscopy images from few examples with Siamese neural networks" (negative staining confirmed from the dataset record: 2 % phosphotungstic acid).
- [x] **Abstract**: one unstructured paragraph, at most 250 words (248, counted on the PDF by `verify_paper.py`).
- [x] **Keywords**: 7 (limit 1–7).
- [x] **Highlights**: `highlights.txt`, 5 bullets, at most 85 characters each.
- [x] **Affiliation with postal address**: Independent Researcher, Shahjadpur, Dhaka 1212, Bangladesh.
- [x] **Appendix numbering** per appendix (Table A.1, Fig. B.1), as the guide asks.
- [x] **Microscopy content**: acquisition and staining in Methods 2.1; new Results subsection "Which rare viruses are confused" (counts for both classifiers, from `make_numbers_tex.py`, recomputed independently in `verify_paper.py`; descriptive, test split) and a Discussion paragraph relating them to virus family and particle appearance, with ICTV taxonomy references (all four checked on ictv.global and Crossref).
- [x] **Declarations**: competing interests, funding, CRediT, data and code availability, generative-AI statement.
- [x] **Code link**: https://github.com/hrid0yyy/tem-virus-siamese-fewshot (public, MIT).
- [x] **Cover letter** rewritten for Micron (`cover_letter.pdf`); discloses the SSRN preprint (abstract ID 7664934).
- [x] **Official template**: Elsevier's elsarticle class, v3.4 (2024/04/04, the current release), generated from the official `elsarticle.dtx` by `make_elsarticle_cls.py` and shipped in the bundle; `verify_paper.py` checks the version in `main.log`.
- [x] **References**: DOIs added for every entry that has one (checked on Crossref / doi.org); the dataset is marked "[dataset]"; the SSRN preprint is disclosed in the cover letter. Efron and Tibshirani (1993) has no DOI for the 1993 edition; NeurIPS/ICLR papers have none.
- [x] **Tables**: notes placed below the table body, no vertical rules; every figure and table is cited in the text (checked by `verify_paper.py`).
- [x] **Abbreviations**: macro-F1 defined in the abstract.
- [x] **Generative-AI statement** names the analysis code, the figure-drawing code and the text. The figures are plots and diagrams drawn by code from the data; no image was generated or altered by a generative-AI tool, so no caption disclosure is needed.
- [x] **No graphical abstract** required. **No main-text word limit.** References may be in any consistent style at submission (Micron converts to author–year after acceptance).

## For you
- [ ] **Final read** of the whole paper, especially the new abstract, the new Results subsection 3.5, the Discussion paragraph ("The confusions are consistent with ...") and the cover letter.
- [ ] Confirm the **SSRN preprint** status (the cover letter says an earlier version "was posted" there, which is true either way).
- [ ] **Declarations tool**: Micron asks you to fill in Elsevier's online declaration-of-interest tool and upload the Word (.docx) file it produces. Choose "I have nothing to declare".

## Rebuild before uploading
`python make_numbers_tex.py --mode fp32`, `tectonic -X compile main.tex --keep-intermediates` (writes the fresh `main.bbl` that is uploaded), `tectonic cover_letter.tex`, `python verify_paper.py` (must print ALL CHECKS PASSED), then `python make_submission.py`.

## Upload to Editorial Manager (all files are in `submission_micron/`)
- **Manuscript (LaTeX source)**: `main.tex`, `numbers.tex`, `references.bib`, `main.bbl`, `elsarticle-num.bst`, `elsarticle.cls`. A PDF is *not* accepted as the source file; `main.pdf` is only for your reference, upload it as well only if the system asks for a PDF.
- **Figures**, each as a separate file: `Figure_1.pdf` (dataset), `Figure_2.pdf` (architecture), `Figure_3.pdf` (confusions), `Figure_4.pdf` (K-shot), `Figure_B1.pdf` (training curves).
- **Highlights**: `highlights.txt`.
- **Cover letter**: `cover_letter.pdf` (or paste its text if there is only a text box).
- **Declaration of interest**: the .docx from Elsevier's declarations tool.
