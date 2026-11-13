# Data required by this checkpoint

The input is a pair of historical triangular-face and topology caches for each CAD model. The repository contains sample identities and labels, not raw geometry or cached features.

```text
face-root/                    topo-root/
  bearing/bearing_0788.bin      bearing/bearing_0788.bin
  bolt/bolt_0385.bin            bolt/bolt_0385.bin
  ...                          ...
```

- `datasplit_relative.json`: historical train, validation and test entries in original order.
- `admission.json`: counts and excluded identities for each split.
- `excluded_test_face_paths.json`: 232 excluded test entries; the evaluator verifies the remaining 794 and their order.
- `class_names.json`: ten labels, with bolt and screw separate and no spring class.

The [upstream dataset page](https://github.com/Qiang-Zou/BRT/blob/main/DATASET.md) provides TMCAD v1 and v2. V2 merges bolt/screw and adds spring; its class semantics do not match this checkpoint. The historical split has v1-style labels, but its exact equivalence to the author's v1 processed sample list has not been established.

The [upstream preprocessing instructions](https://github.com/Qiang-Zou/BRT#4usage) describe face/topology extraction from STEP. Running the latest preprocessing and generating a new random split is not a verified way to reproduce these historical caches or the reported score. Do not run the split-generation command when attempting to reproduce this release's evaluation; use the supplied membership and order.

Face caches contain `nodes`, `in_mask`, `points` and `tri_normals`. Topology caches contain edge control points, wire-to-edge references, face-to-wire references and face adjacency. The bundled loader applies the original normalization, truncation and padding. Caches are serialized objects read with PyTorch; use matching data from a trusted source.

A public download location for the matching legacy caches has not been supplied. This limitation is explicit rather than replaced with a claim that arbitrary regenerated data will yield 83.88%.
