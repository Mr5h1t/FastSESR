# Local FastSESR improvements and experiments

This branch contains optional improvements and recorded experiments:

1. **On-demand triangle construction:** construct only the two triangle candidates
   consumed by extraction, preserving the original score sort and output faces.
   The 40-shape ABC experiment reduced extraction time by 63.94% and extraction
   extra peak GPU allocation by 81.93%. Full reconstruction time fell by 2.49%;
   full-pipeline peak allocation was unchanged. All 40 output meshes were identical.
2. **Configurable face thresholds:** use the constructor confidence thresholds
   instead of hard-coded constants; expose them in the reconstruction CLI.
   A 16-shape development search selected top-1=0.8, top-2=0.6, angle=120 degrees.
   On 40 separate test shapes, mean non-manifold edges fell by 33.41%, while
   boundary edges increased. F1 and CD differences were small; there is no
   demonstrated stable accuracy improvement. Defaults remain 0.8/0.5/120.

3. **Planar overlap repair:** optional CPU postprocessing that retains vertex
   positions and checks coverage, seams and global indexed edge counts. The
   independent 38-file subset reduced mean non-manifold edges by 50.04% and
   boundary edges by 15.21%. Sampled CD differences include zero in their paired
   intervals; continuous-surface CD1 increased by 0.93%, and runtime increased
   from 0.367 to 1.007 seconds. Default behavior remains unchanged.
4. **Geometry-adaptive threshold trial:** eleven development configurations did
   not improve accuracy within the constraints. The selected control retained
   weight zero. The negative trial and its executed source snapshots are saved.

Recorded reports and evidence:

- [Performance report](results/2026-10-09/full/REPORT.md)
- [Performance comparison CSV](results/2026-10-09/full/comparison.csv)
- [Threshold report](results/2026-10-10/thresholds/REPORT.md)
- [Threshold comparison CSV](results/2026-10-10/thresholds/comparison.csv)
- [Planar repair report](results/2026-10-10/planar_repair/REPORT.md)
- [Deduplicated planar comparison](results/2026-10-10/planar_repair/heldout_comparison.csv)
- [Adaptive negative result](results/2026-10-10/adaptive_quality/REPORT.md)

The reports, raw metric/timing CSVs, figures, sample selections and source/weight
hash manifests are versioned. Local point clouds and reconstructed mesh outputs
are excluded. Upstream's already tracked pretrained models remain in repository
history; this change adds no new model weights.

## Run tests

From the project root in an environment satisfying the project dependencies:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python benchmarks/test_extract_face.py
python benchmarks/test_planar_repair.py
```

The exact pre-optimization extractor is retained in
`baselines/extract_face_before.py` for differential tests. CUDA-specific coverage
runs when CUDA is available.

## Reproduce experiments

The reports record the original server commands and dataset paths. On another
machine, place the ABC point clouds and ground-truth meshes under the layout in
[the main README](../README.md), and update sample paths in a **copy** of
`results/2026-10-09/selected_abc_test.json` or
`results/2026-10-10/thresholds/protocol.json`. Preserve the recorded manifests as
historical evidence. Do not mix the development and test sample lists.

For the threshold experiment, first reproduce the speed benchmark so that the
original output meshes used by its compatibility check exist at
`results/2026-10-09/full/original/`. Both experiment scripts initialize weights
through the official reconstruction entry point and verify the active pretrained
parameters. See the individual reports for all arguments and evaluation limits.

These are personal experiment results, not upstream claims or default changes.

The planar experiment originally selected different CAD document names. A later
byte-hash audit found two inputs duplicated across experiment lists. The formal
planar report excludes those files using a metric-independent rule and retains
all 40 original records. Earlier reports remain historical records; distinct
file names alone should not be interpreted as proof of distinct geometry.
