# WPRO LaTeX Draft

Main replanned file:

```text
paper/wpro_infocom_replanned.tex
```

Earlier draft kept for comparison:

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

Additional system-design figure:

```text
paper/Figures/fig_system_future_coupling.pdf
```

PDF preview generated without a local LaTeX compiler:

```text
paper/wpro_infocom_replanned_preview.pdf
```

Recommended build command from the repository root:

```powershell
cd paper
pdflatex wpro_infocom_replanned.tex
bibtex wpro_infocom_replanned
pdflatex wpro_infocom_replanned.tex
pdflatex wpro_infocom_replanned.tex
```

This local Windows environment currently does not expose `pdflatex`, `xelatex`, or `latexmk` in PATH, so the LaTeX source was syntax-checked for citation-key consistency but not locally rendered. A ReportLab PDF preview was generated for visual review. The TeX should compile on Overleaf or a TeX Live/MiKTeX installation with IEEEtran, algorithm, algpseudocode, and standard AMS packages.

Important note: the current figures under `paper_artifacts/figure_data` are draft figure data for layout and paper-story review. Before submission, replace the CSV files with final held-out trace results and regenerate figures using:

```powershell
py generate_wpro_paper_figures.py --data-dir paper_artifacts\figure_data --output-dir paper_artifacts\figures
```
