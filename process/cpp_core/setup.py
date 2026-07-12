import os

import pybind11
from setuptools import Extension, setup

conda = os.environ.get("CONDA_PREFIX")
if not conda:
    raise RuntimeError("build inside the tok conda env (CONDA_PREFIX unset)")

ext = Extension(
    "brt_core",
    ["brt_core.cpp"],
    include_dirs=[
        pybind11.get_include(),
        os.path.join(conda, "include", "opencascade"),
    ],
    library_dirs=[os.path.join(conda, "lib")],
    runtime_library_dirs=[os.path.join(conda, "lib")],
    libraries=[
        "TKernel",
        "TKMath",
        "TKG2d",
        "TKG3d",
        "TKGeomBase",
        "TKGeomAlgo",
        "TKBRep",
        "TKTopAlgo",
    ],
    extra_compile_args=["-O3", "-std=c++17", "-ffp-contract=off", "-fvisibility=hidden"],
    language="c++",
)

setup(name="brt_core", version="0.1", ext_modules=[ext])
