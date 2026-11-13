"""Drop-in C++ path for convertFaceToTriangles (method 8).

convertFaceToTriangles_cpp mirrors process/solid_to_triangles2.py:
convertFaceToTriangles but runs the quadtree subdivision, boundary cutting,
triangle collection and control point machinery in the brt_core pybind11
module. On any per-face error from the C++ core it logs and falls back to the
stock Python implementation for that face. See README.md for build and
activation instructions; brt_core must be built next to this file.

The parent process/ directory is added to sys.path automatically, matching
the flat-module layout the rest of the pipeline uses.
"""

import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import brt_core
from OCC.Core.GeomConvert import GeomConvert_BSplineSurfaceToBezierSurface as Converter
from OCC.Core.TColStd import TColStd_Array1OfReal as ArrReal

from solid_to_triangles2 import (
    convertFaceToTriangles,
    doKnotInsertion,
    getNURBS,
    pcurve,
)
from utils import bezier2

# The exact uv grid getControlPointsFromApproximation uses: 9 fixed rows plus
# the module-import-time random 100 rows living in the function's default
# argument. Reading the live default keeps the C++ path bitwise aligned with
# the Python path within this process.
_UVS_BASE = np.array(
    [[0, 0], [0, 1], [1, 0], [0, 1 / 3], [0, 2 / 3], [1 / 3, 2 / 3],
     [2 / 3, 1 / 3], [2 / 3, 0], [1 / 3, 0]]
)
_UVS_RANDOM = bezier2.getControlPointsFromApproximation.__defaults__[2]
_UVS = np.concatenate([_UVS_BASE, _UVS_RANDOM], axis=0)


def _fit_tail(points):
    """Exact tail of getControlPointsFromApproximation after the surface
    sampling: normalization, lstsq fit (module bn_cache), denormalization."""
    points = np.array(points)
    center, scale = bezier2.getCenterAndScale(points)
    points = points - center
    points = points * scale
    ctrl_pts = bezier2.fit_bezier_surface2(points, _UVS, points)
    ctrl_pts[..., :3] /= scale
    ctrl_pts[..., :3] += center
    return ctrl_pts


def _collect_cpp(face):
    """C++ counterpart of the collection stage of convertFaceToTriangles
    (trim=True): returns (nodes, in_mask, tri_normals, (uknots, vknots))."""
    surface, loc = getNURBS(face)
    doKnotInsertion(surface, num_max_knots=5)

    converter = Converter(surface)
    uNumPatches = converter.NbUPatches()
    vNumPatches = converter.NbVPatches()
    if uNumPatches == 0 or vNumPatches == 0:
        raise RuntimeError("no patches")

    uKnots = ArrReal(1, uNumPatches + 1)
    vKnots = ArrReal(1, vNumPatches + 1)
    converter.UKnots(uKnots)
    converter.VKnots(vKnots)
    uknots = [uKnots[i] for i in range(uNumPatches + 1)]
    vknots = [vKnots[i] for i in range(vNumPatches + 1)]

    crvs = []
    for wire in face.wires():
        for edge in wire.ordered_edges():
            crv, interval = pcurve(face, edge)
            crvs.append((crv, interval))

    tds = face.topods_shape()
    curve_args = [
        (int(crv.this), float(interval[0]), float(interval[1]))
        for crv, interval in crvs
    ]

    res = brt_core.process_face(
        int(tds.this), int(surface.this), uknots, vknots, curve_args, _UVS
    )
    # keep tds / surface / crvs alive until after the call (they own the OCC
    # objects the C++ side borrowed)
    del tds, surface, crvs

    nodes = res["ctrl"]
    for slot in range(len(res["fit_idx"])):
        nodes[res["fit_idx"][slot]] = _fit_tail(res["fitpts"][slot])

    return nodes, res["in_mask"], res["normals"], (uknots, vknots)


def convertFaceToTriangles_cpp(
    face, num_sample_points=256, normalize=True, trim=True,
    rotated_and_normalized=True, **kwargs
):
    if not trim:
        return convertFaceToTriangles(
            face, num_sample_points=num_sample_points, normalize=normalize,
            trim=trim, rotated_and_normalized=rotated_and_normalized, **kwargs
        )

    try:
        nodes, in_mask, tri_normals, (uknots, vknots) = _collect_cpp(face)

        # phase 2: rotation/normalization loop in C++ (mutates nodes in
        # place; returns the last iteration's center/scale, matching the
        # original loop's repeatedly overwritten feature columns)
        if rotated_and_normalized:
            new_feature = np.zeros((len(tri_normals), 7), dtype=tri_normals.dtype)
            new_feature[:, :3] = tri_normals
            center_r, scale_r = brt_core.rotate_normalize(nodes, tri_normals)
            new_feature[:, 3:6] = center_r
            new_feature[:, 6] = scale_r
            tri_normals = new_feature

        # phase 2: randn_uvgrid stage. The RNG stays in Python: exactly one
        # np.random.uniform draw per face, as in the stock "point" call
        # (normal/visibility reuse it via given_uvs), keeping the RNG stream
        # aligned. C++ does the uv interpolation and the surface / normal /
        # classifier evaluations.
        uv_values = np.random.uniform(size=(num_sample_points, 2)).astype(np.float32)
        tds = face.topods_shape()
        sampled = brt_core.sample_face(
            int(tds.this), uv_values,
            uknots[0], uknots[-1], vknots[0], vknots[-1],
        )
        del tds
        points = sampled["points"]
        normals = sampled["normals"]
        visibility_status = sampled["vis"]
    except Exception as e:
        logging.warning(f"[brt_core fallback] {type(e).__name__}: {e}")
        return convertFaceToTriangles(
            face, num_sample_points=num_sample_points, normalize=normalize,
            trim=trim, rotated_and_normalized=rotated_and_normalized, **kwargs
        )

    mask = np.logical_or(visibility_status == 0, visibility_status == 2)

    if normalize:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        bbox = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        bbox = np.array(bbox)

        diag = bbox[1] - bbox[0]
        scale = 2.0 / max(diag[0], diag[1], diag[2])
        center = 0.5 * (bbox[0] + bbox[1])

        points -= center
        points *= scale

        nodes[..., :3] -= center
        nodes[..., :3] *= scale
    else:
        scale = 1.0
        center = np.zeros(3)

    points = torch.from_numpy(points)
    uv_values = torch.from_numpy(uv_values)
    vis_mask = torch.from_numpy(mask)
    scale = torch.tensor(scale)
    normals = torch.tensor(normals)

    nodes = torch.from_numpy(nodes)
    in_mask = torch.from_numpy(in_mask)
    tri_normals = torch.from_numpy(tri_normals)

    return nodes, in_mask, tri_normals, points, normals, vis_mask, uv_values, scale
