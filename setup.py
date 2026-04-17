"""Setup script for pydnn — builds the C++/CUDA extension via CMake."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


HERE = Path(__file__).parent.resolve()


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not isinstance(ext, CMakeExtension):
            return super().build_extension(ext)

        # Resolve where setuptools expects the final compiled module to land.
        # e.g. build/lib.linux-x86_64-cpython-3XX/pydnn/_dnn_core.*.so
        ext_fullpath = Path(self.get_ext_fullpath(ext.name)).resolve()
        package_dir = ext_fullpath.parent
        package_dir.mkdir(parents=True, exist_ok=True)

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # Locate pybind11 installed in the build environment so CMake's
        # find_package(pybind11 CONFIG) in the root CMakeLists can find it.
        import pybind11  # noqa: WPS433  (build-time only)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={package_dir}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DDNN_BUILD_PYTHON=ON",
            "-DDNN_BUILD_TESTS=OFF",
            "-DDNN_BUILD_EXAMPLES=OFF",
        ]

        # Pass CUDA toggle through verbatim when the user sets it; otherwise
        # let CMake's check_language(CUDA) auto-detect nvcc.
        if "DNN_ENABLE_CUDA" in os.environ:
            cmake_args.append(f"-DDNN_ENABLE_CUDA={os.environ['DNN_ENABLE_CUDA']}")

        build_args = ["--config", cfg]

        # Windows multi-config generator needs per-config output dir,
        # and x64 must be explicit on MSVC.
        if sys.platform.startswith("win"):
            cmake_args.append(
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={package_dir}"
            )
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
        else:
            parallel = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(
                os.cpu_count() or 1
            )
            build_args += ["--parallel", parallel]

        # Allow power users to append extra flags via env var.
        if "CMAKE_ARGS" in os.environ:
            cmake_args += shlex.split(os.environ["CMAKE_ARGS"])

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ["cmake", ext.sourcedir, *cmake_args], cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", *build_args], cwd=build_temp
        )


long_description = ""
readme_path = HERE / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")


setup(
    name="pydnn",
    version="0.0.1",
    author="DNN Team",
    author_email="dnn@example.com",
    description="Dynamic Neural Network with automatic architecture adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dnn/dynamic_nn",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.5.0", "plotly>=5.0.0"],
        "image": ["pillow>=9.0.0"],
        "video": ["opencv-python>=4.5.0"],
        "audio": ["librosa>=0.9.0"],
        "all": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
            "pillow>=9.0.0",
            "opencv-python>=4.5.0",
            "librosa>=0.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pybind11>=2.10.0",
        ],
    },
    ext_modules=[CMakeExtension("pydnn._dnn_core", sourcedir=str(HERE))],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
