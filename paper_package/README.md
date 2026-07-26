# Paper package — Compressed Traffic Assignment, draft v3.4

Self-contained LaTeX source. Drop the whole folder into Overleaf, or compile locally.

## Contents

    main.tex                   the manuscript
    ref.bib                    bibliography
    fig/                       the four figures main.tex uses
      fig_compression_pipeline.pdf   conceptual, Section 3
      fig_speedup_mechanisms.pdf     grid speedup vs pool depth; rank sweep
      fig_paired.pdf                 paired comparisons with the speed noise floor
      fig_3d.pdf                     gap surfaces over the grid design
    main_v3.4_reference.pdf    reference build, to check your compile matches

## Compile

    pdflatex main
    bibtex   main
    pdflatex main
    pdflatex main

Expected: 31 pages, no errors, no undefined references.

## Conventions in this draft

- **The numerical section is set in black.** It is a full rewrite rather than a redline, so
  marking it would have marked everything. Earlier sections retain their v2-to-v3 red.
- **Red now means a note to co-authors.** The `\conote` and `\conoteY` macros are defined just
  above `\begin{document}`; there are seven calls. Delete the macro definitions and the calls
  before submission.
- **The version stamp** under the title comes from `\docver`, `\docdate` and `\doccommit`.
  Bump `\docver` on every circulated build so two PDFs are never confused — an earlier
  all-red build was reviewed by mistake because the filename and title were identical.

## Where the numbers come from

Every figure and table traces to a CSV in the companion repository
(`asu-trans-ai-lab/CompressedTAP`, branch `version_3`):

- `campaign_weighted/results/` — the flow-weighted campaign
- `repro_v3/results/` — the earlier unweighted campaign, retained for comparison

`campaign_weighted/REPRODUCIBILITY.md` maps command to CSV to table or figure in both
directions, and states the precision policy: accuracy figures are exact, timing carries a
1.45x noise floor, and speed differences below that floor are not claimed.
