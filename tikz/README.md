# `tikz/` — tensor-network diagrams for the paper

TikZ/LaTeX sources for the paper's tensor-network figures.

## Files
- **`tensor-network.sty`** — the tensor-network drawing macros (tensors, legs/bonds, isometries,
  contractions). Vendored from the upstream library below.
- **`rmps_disentangled_column_method.{tex,png}`** — the disentangled-column method figure (the exp10
  mechanism picture: sketch sweep + Moses disentangler on the composite bond), built with that `.sty`.

## Upstream — for drawing more paper figures
**https://github.com/mptoolkit/tikz-diagrams** (GPL-3.0) — a collection of example tensor-network
diagrams built on `tensor-network.sty`. Our `tensor-network.sty` is a vendored copy of theirs. Their
`main.tex` / `example.tex` are the worked examples to copy from when drawing new figures for the paper
(columns, Moses moves, MPS/MPO/PEPS, isoTNS, sketches, the disentangled column). Extend that `.sty`
rather than re-rolling macros. GPL-3.0 is copyleft — keep the attribution on figures built with it.

## Build
```bash
pdflatex rmps_disentangled_column_method.tex     # standalone
```
or `\input{tikz/rmps_disentangled_column_method}` into the paper source.
