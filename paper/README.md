# WPRO LaTeX Draft

Main file:

```text
paper/wpro_infocom_draft.tex
```

References:

```text
paper/references.bib
```

The draft uses the vector PDF figures generated from CSV data under:

```text
paper_artifacts/figures/
```

Recommended build command from the repository root:

```powershell
cd paper
pdflatex wpro_infocom_draft.tex
bibtex wpro_infocom_draft
pdflatex wpro_infocom_draft.tex
pdflatex wpro_infocom_draft.tex
```

This local Windows environment currently does not expose `pdflatex`, `xelatex`, or `latexmk` in PATH, so the source was syntax-checked for citation-key consistency but not locally rendered. It should compile on Overleaf or a TeX Live/MiKTeX installation with IEEEtran, algorithm, algpseudocode, tikz, and standard AMS packages.

Important note: the current figures under `paper_artifacts/figure_data` are draft figure data for layout and paper-story review. Before submission, replace the CSV files with final held-out trace results and regenerate figures using:

```powershell
py generate_wpro_paper_figures.py --data-dir paper_artifacts\figure_data --output-dir paper_artifacts\figures
```
