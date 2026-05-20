import os
from pathlib import Path

from setuptools import setup
from torch.utils import cpp_extension


REPO_ROOT = Path(__file__).resolve().parent
CSRC = REPO_ROOT / "csrc"


def _can_use_cuda() -> bool:
    if os.environ.get("ENN_FORCE_CPU"):
        return False
    return cpp_extension.CUDA_HOME is not None


def _gather_sources(use_cuda: bool):
    cpp_sources = sorted(str(p.relative_to(REPO_ROOT))
                          for p in CSRC.rglob("*.cpp"))
    cu_sources = sorted(str(p.relative_to(REPO_ROOT))
                         for p in CSRC.rglob("*.cu")) if use_cuda else []
    return cpp_sources + cu_sources


def build_extension():
    use_cuda = _can_use_cuda()
    sources = _gather_sources(use_cuda)
    include_dirs = [str(CSRC / "include")]
    extra_compile_args = {
        "cxx": ["-O3", "-std=c++17"],
    }
    if use_cuda:
        extra_compile_args["nvcc"] = ["-O3", "--std=c++17"]
        ext_cls = cpp_extension.CUDAExtension
    else:
        ext_cls = cpp_extension.CppExtension

    return ext_cls(
        name="elasticneuralnetwork._enn_core",
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
    )


setup(
    ext_modules=[build_extension()],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
