# N01 readout-aware waveform calibration: reproducible negative result

The S research candidate improves waveform and morphology distances but fails native protection in all six evaluated conditions. `scientific_acceptance=false`.

Start with [the manuscript draft](MANUSCRIPT_EN.md), [six figure captions](FIGURE_CAPTIONS_EN.md), [related work](RELATED_WORK.md) or [data definitions](data/README.md). PNG and editable SVG figures are in `figures/`.

Run the [one-event synthetic example](public_demo/README.md) with the recorded direct dependency versions in `runtime/requirements-public-demo.txt`. Rebuild the figures with `python -B REBUILD_FIGURES.py`. Both use only the included synthetic or aggregate data. `MANIFEST.json` records each included member's byte size and SHA256.

This shareable bundle contains no individual SOURCE/REAL waveform, parent labels, neural checkpoint or fitted calibration template. It reproduces the constrained mathematics and figures. Full neural inference and population evaluation require the authorized private environment, frozen weights and original data dependencies. No standalone full-model portability is claimed.

The planned fair P0/P1/CPU-Net/N01 comparison, full-pipeline held-out validation, repeated-parent kernel study, new-endpoint C2ST and alternative tail-window evaluation have not run. Their score fields are blank with explicit status.
