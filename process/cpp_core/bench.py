"""End-to-end benchmark: stock Python path vs brt_core path.

Runs solid_to_triangles2.main (method 8, no_label, no_random_name) on an
input folder, optionally monkeypatching convertFaceToTriangles with the C++
wrapper. Output dirs are wiped first (the pipeline skips existing outputs).

Usage:
    python bench.py <input_dir> <output_dir> --mode {py,cpp} [--procs N]
"""

import argparse
import os
import shutil
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # process/
sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["py", "cpp"], required=True)
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    import solid_to_triangles2 as stt

    if args.mode == "cpp":
        import brt_wrapper

        stt.convertFaceToTriangles = brt_wrapper.convertFaceToTriangles_cpp

    n_parts = len(
        [f for f in os.listdir(args.input) if f.endswith(("step", "stp"))]
    )
    t0 = time.perf_counter()
    stt.main(
        [
            args.input,
            args.output,
            "--method",
            "8",
            "--no_label",
            "--no_random_name",
            "--num_processes",
            str(args.procs),
        ]
    )
    dt = time.perf_counter() - t0
    n_out = len([f for f in os.listdir(args.output) if f.endswith(".bin")])
    print(
        f"[bench] mode={args.mode} procs={args.procs} parts={n_parts} "
        f"outputs={n_out} time={dt:.2f}s s/part={dt / max(n_parts, 1):.3f}"
    )


if __name__ == "__main__":
    main()
