# cpp_core: optional C++ fast path for face tokenization

`brt_core` is a pybind11 module that replaces the per-face hot path of
`process/solid_to_triangles2.py` (method 8): the quadtree subdivision,
boundary-rectangle cutting, triangle collection, Bezier control-point
machinery, the per-triangle rotation/normalization loop, and the 256-point
uv sampling stage. Everything else (STEP loading, NURBS conversion, pcurve
extraction, RNG, least-squares fitting, torch serialization) stays in the
existing Python code.

It is **opt-in only**: nothing in the default code path changes. If the
module is not built or not activated, the pipeline behaves exactly as
before.

## Design and equivalence

- The module links against the *same* OCCT shared libraries that
  pythonocc-core wraps, and borrows the live OCC objects (face, surfaces,
  2D trimming curves) by pointer for the duration of each call. Every
  OCC-derived quantity (intersections, projections, classifier states,
  Bezier patch poles, surface evaluations) is therefore produced by the
  identical OCC code on the identical objects, so all discrete decisions
  (splits, in/out masks, triangle counts) match the Python implementation
  exactly, and the pipeline's behavioral quirks are reproduced faithfully.
- Random number generation stays in Python (`np.random`), one uniform draw
  per face exactly like the stock path, so the RNG stream is unchanged.
- The boundary-triangle least-squares fit stays in Python (fed with
  C++-evaluated sample points), so fitted control points are bit-identical.
- `equiv_test.py` compares the two implementations per face: triangle
  counts, masks, sampled points/normals/visibility and uv values are
  bitwise equal; remaining control-point differences are at the 1e-14 level
  (matmul summation order only).

## Build

Inside the conda environment that provides pythonocc-core (the OCCT
headers and libraries ship with it):

```bash
pip install pybind11
cd process/cpp_core
python setup.py build_ext --inplace
```

This produces `brt_core.cpython-*.so` next to the sources. The extension
links `TKernel TKMath TKG2d TKG3d TKGeomBase TKGeomAlgo TKBRep TKTopAlgo`
from `$CONDA_PREFIX/lib` with an rpath into the environment, and is
compiled with `-ffp-contract=off` so scalar arithmetic rounds exactly like
numpy. Tested with pythonocc-core 7.7.2 / OCCT 7.7.2, numpy 1.26, python
3.10 on linux-x86_64.

## Activation

`brt_wrapper.convertFaceToTriangles_cpp` is a drop-in replacement for
`convertFaceToTriangles` with the same signature and outputs. On any error
from the C++ core it logs `[brt_core fallback]` and re-runs the stock
Python implementation for that face, so worst-case behavior is the
original behavior.

```python
import sys
sys.path.insert(0, "process")          # as usual for this repo
sys.path.insert(0, "process/cpp_core")

import solid_to_triangles2 as stt
import brt_wrapper

stt.convertFaceToTriangles = brt_wrapper.convertFaceToTriangles_cpp
stt.main([input_dir, output_dir, "--method", "8", "--no_label",
          "--no_random_name", "--num_processes", "32"])
```

`bench.py` wraps exactly this for timing:

```bash
python bench.py <step_dir> <out_dir> --mode cpp --procs 32   # C++ path
python bench.py <step_dir> <out_dir> --mode py  --procs 32   # stock path
```

## Verification

```bash
python equiv_test.py <folder of .step files>
```

Runs both implementations on every face of every part in the folder and
prints a per-part pass/fail table (triangle counts, masks, control points,
normals, plus a seeded full-pipeline comparison of all eight
convertFaceToTriangles outputs).
