# Frozen prediction models

Each `<dataset>.pt` contains the calibrated prediction state used by the current KL_H4 four-probe implementation. All 173 tensors, including output calibration buffers, preserve the source dtype and values exactly. Only the effective calibrated state is retained.

Portable checkpoint fields:

```text
format        eis-fairpath-teacher-v1
dataset       full graph identifier
architecture  complete model construction arguments
sampler       native fixed-support sampling configuration
state_dict    calibrated model parameters and buffers
```

`manifest.json` records the portable files' hashes and sizes, the source checkpoint hashes, and the matching complete graph hashes. Loading uses `torch.load(..., weights_only=True)`, exact state-dict matching, dtype-preserving assignment, CPU placement, evaluation mode, and frozen parameters.

These prediction models supply the importance probes. The explanation mask is initialized and optimized independently for every query. To use another trained model of the same architecture, save its calibrated state and construction/sampling arguments in the portable format and pass its path with `--checkpoint`.
