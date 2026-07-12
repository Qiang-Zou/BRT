"""Equivalence harness: Python vs brt_core.

Phase 1 stage comparison, per face (collection): triangle count, in/out
mask, control points, normals.

Phase 2 full-pipeline comparison, per face: np.random is seeded identically
before the stock convertFaceToTriangles and the C++
convertFaceToTriangles_cpp (both consume exactly one uniform draw per face,
so the RNG streams align) and ALL eight outputs are compared: nodes,
in_mask, tri_normals (the 7-column feature), points, normals, vis_mask,
uv_values, scale. normalize=False as in production (build_triangles).

Both paths run in the SAME process so they share the module-import-time
random uv fit grid (import-time random uv grid). Faces where both implementations raise are
counted as matching failures.

Usage (inside the pythonocc conda env, after building brt_core):
    python equiv_test.py <folder of .step files>
"""

import glob
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # process/
sys.path.insert(0, _HERE)

import numpy as np
from occwl.compound import Compound
from occwl.graph import face_adjacency
from OCC.Core.GeomConvert import GeomConvert_BSplineSurfaceToBezierSurface as Converter
from OCC.Core.TColStd import TColStd_Array1OfReal as ArrReal

import solid_to_triangles2 as stt
import triangles3
from triangles3 import CollectTrisInLine, HandleLeaves, Rectangle, splitRectangle

import brt_wrapper


def python_collect(face):
    """Collection stage of convertFaceToTriangles (trim=True), verbatim."""
    surface, loc = stt.getNURBS(face)
    stt.doKnotInsertion(surface, num_max_knots=5)

    converter = Converter(surface)
    uNumPatches = converter.NbUPatches()
    vNumPatches = converter.NbVPatches()
    if uNumPatches == 0 or vNumPatches == 0:
        raise RuntimeError("no patches")

    uKnots = ArrReal(1, uNumPatches + 1)
    vKnots = ArrReal(1, vNumPatches + 1)
    converter.UKnots(uKnots)
    converter.VKnots(vKnots)

    rects = []
    for u in range(uNumPatches):
        for v in range(vNumPatches):
            rect = Rectangle()
            rect.points = [
                (uKnots[u], vKnots[v]),
                (uKnots[u + 1], vKnots[v]),
                (uKnots[u], vKnots[v + 1]),
                (uKnots[u + 1], vKnots[v + 1]),
            ]
            rects.append(rect)

    crvs = []
    for wire in face.wires():
        for edge in wire.ordered_edges():
            crv, interval = stt.pcurve(face, edge)
            crvs.append((crv, interval))

    tris = []
    with triangles3.suppress_subdivsion_err():
        for rect in rects:
            splitRectangle(face, rect, crvs, max_split=5)
            HandleLeaves(face, rect, surface, loc)
            CollectTrisInLine(rect, tris, face, surface, loc)

    def getTriNormal(tri):
        x = (tri.v1[0] + tri.v2[0] + tri.v3[0]) / 3
        y = (tri.v1[1] + tri.v2[1] + tri.v3[1]) / 3
        return face.normal([x, y])

    nodes = [t.control_points if type(t) is not tuple else t[1].control_points for t in tris]
    nodes = np.stack(nodes)
    in_mask = np.array([type(t) is not tuple for t in tris])
    tri_normals = [getTriNormal(t) if type(t) is not tuple else getTriNormal(t[1]) for t in tris]
    tri_normals = np.stack(tri_normals)
    return nodes, in_mask, tri_normals


def main():
    if len(sys.argv) < 2:
        print("usage: python equiv_test.py <folder of .step files>")
        sys.exit(2)
    input_dir = sys.argv[1]
    parts = sorted(glob.glob(os.path.join(input_dir, "*.st*p")))
    if not parts:
        print(f"no step files in {input_dir}")
        sys.exit(1)

    header = (
        f"{'part':<16}{'faces':>6}{'tris_py':>9}{'tris_cpp':>9}{'mask':>6}"
        f"{'max|dctrl|':>13}{'max|dnrm|':>12}{'max|dnodes|':>13}{'max|dfeat|':>12}"
        f"{'sampled':>9}{'bothfail':>9}  {'status'}"
    )
    print(header)
    print("-" * len(header))

    all_pass = True
    grand_ctrl = 0.0
    grand_nrm = 0.0
    grand_nodes = 0.0
    grand_feat = 0.0

    for part in parts:
        name = os.path.basename(part)
        try:
            comp = Compound.load_from_step(part)
            solid = next(comp.solids())
            graph = face_adjacency(solid)
        except Exception as e:
            print(f"{name:<16} LOAD ERROR: {e}")
            all_pass = False
            continue

        nfaces = 0
        tris_py = 0
        tris_cpp = 0
        mask_ok = True
        count_ok = True
        sampled_ok = True
        max_ctrl = 0.0
        max_nrm = 0.0
        max_nodes = 0.0
        max_feat = 0.0
        both_fail = 0
        mismatch_msgs = []

        for face_idx in graph.nodes:
            face = graph.nodes[face_idx]["face"]
            nfaces += 1

            # ---------------- phase 1: collection stage ----------------
            py_res, py_err = None, None
            cpp_res, cpp_err = None, None
            try:
                py_res = python_collect(face)
            except Exception as e:
                py_err = f"{type(e).__name__}: {e}"
            try:
                cpp_res = brt_wrapper._collect_cpp(face)[:3]
            except Exception as e:
                cpp_err = f"{type(e).__name__}: {e}"

            if py_res is None or cpp_res is None:
                if py_res is None and cpp_res is None:
                    both_fail += 1
                    continue
                mismatch_msgs.append(
                    f"face {face_idx}: py_err={py_err} cpp_err={cpp_err}"
                )
                count_ok = False
                continue

            n_py, m_py, nn_py = py_res
            n_cpp, m_cpp, nn_cpp = cpp_res
            tris_py += len(m_py)
            tris_cpp += len(m_cpp)

            if len(m_py) != len(m_cpp):
                count_ok = False
                mismatch_msgs.append(
                    f"face {face_idx}: tri count {len(m_py)} vs {len(m_cpp)}"
                )
                continue
            if not np.array_equal(m_py, m_cpp):
                mask_ok = False
                mismatch_msgs.append(f"face {face_idx}: mask mismatch")
                continue
            max_ctrl = max(max_ctrl, float(np.max(np.abs(n_py - n_cpp))))
            max_nrm = max(max_nrm, float(np.max(np.abs(nn_py - nn_cpp))))

            # ------------- phase 2: full pipeline, seeded RNG -------------
            seed = (hash(name) ^ face_idx) & 0x7FFFFFFF
            full = {}
            for tag, fn in (
                ("py", stt.convertFaceToTriangles),
                ("cpp", brt_wrapper.convertFaceToTriangles_cpp),
            ):
                np.random.seed(seed)
                try:
                    full[tag] = fn(face, normalize=False)
                except Exception as e:
                    full[tag] = f"{type(e).__name__}: {e}"
            fail_py = isinstance(full["py"], str)
            fail_cpp = isinstance(full["cpp"], str)
            if fail_py or fail_cpp:
                if fail_py and fail_cpp:
                    continue  # matching failure (collection already passed)
                count_ok = False
                mismatch_msgs.append(
                    f"face {face_idx}: full py={full['py'] if fail_py else 'ok'} "
                    f"cpp={full['cpp'] if fail_cpp else 'ok'}"
                )
                continue
            (nod_p, im_p, tf_p, pt_p, no_p, vm_p, uv_p, sc_p) = full["py"]
            (nod_c, im_c, tf_c, pt_c, no_c, vm_c, uv_c, sc_c) = full["cpp"]
            if nod_p.shape != nod_c.shape or not np.array_equal(
                im_p.numpy(), im_c.numpy()
            ):
                count_ok = False
                mismatch_msgs.append(f"face {face_idx}: full shape/mask mismatch")
                continue
            max_nodes = max(
                max_nodes, float(np.max(np.abs(nod_p.numpy() - nod_c.numpy())))
            )
            max_feat = max(
                max_feat, float(np.max(np.abs(tf_p.numpy() - tf_c.numpy())))
            )
            samp_exact = (
                np.array_equal(pt_p.numpy(), pt_c.numpy())
                and np.array_equal(no_p.numpy(), no_c.numpy())
                and np.array_equal(vm_p.numpy(), vm_c.numpy())
                and np.array_equal(uv_p.numpy(), uv_c.numpy())
                and float(sc_p) == float(sc_c)
            )
            if not samp_exact:
                sampled_ok = False
                mismatch_msgs.append(f"face {face_idx}: sampled outputs differ")

        ok = (
            count_ok and mask_ok and sampled_ok
            and max_ctrl <= 1e-9 and max_nrm <= 1e-9
            and max_nodes <= 1e-9 and max_feat <= 1e-9
        )
        all_pass = all_pass and ok
        grand_ctrl = max(grand_ctrl, max_ctrl)
        grand_nrm = max(grand_nrm, max_nrm)
        grand_nodes = max(grand_nodes, max_nodes)
        grand_feat = max(grand_feat, max_feat)
        print(
            f"{name:<16}{nfaces:>6}{tris_py:>9}{tris_cpp:>9}"
            f"{('ok' if mask_ok else 'FAIL'):>6}{max_ctrl:>13.3e}{max_nrm:>12.3e}"
            f"{max_nodes:>13.3e}{max_feat:>12.3e}"
            f"{('bit' if sampled_ok else 'FAIL'):>9}"
            f"{both_fail:>9}  {'PASS' if ok else 'FAIL'}"
        )
        for msg in mismatch_msgs:
            print(f"    !! {msg}")

    print("-" * len(header))
    print(
        f"overall max|dctrl| = {grand_ctrl:.3e}   max|dnormal| = {grand_nrm:.3e}"
        f"   max|dnodes| = {grand_nodes:.3e}   max|dfeat| = {grand_feat:.3e}"
    )
    print("ALL PASS" if all_pass else "FAILURES PRESENT")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
