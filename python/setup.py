"""
Setup script for pydnn - Dynamic Neural Network Python package.
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """CMake-based extension."""

    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    """Build extension using CMake."""

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake configuration arguments
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DBUILD_PYTHON_BINDINGS=ON",
            "-DBUILD_TESTS=OFF",
        ]

        build_args = []

        # Platform-specific settings
        if sys.platform.startswith("win"):
            cmake_args += [
                f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}",
            ]
            if sys.maxsize > 2**32:
                cmake_args += ["-A", "x64"]
            build_args += ["--config", cfg]
        else:
            build_args += ["--", "-j4"]

        # Set build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        # Run CMake
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp
        )


# Read long description
this_directory = Path(__file__).parent
long_description = ""
readme_path = this_directory.parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="pydnn",
    version="1.0.0",
    author="DNN Team",
    author_email="dnn@example.com",
    description="Dynamic Neural Network with automatic architecture adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dnn/dynamic_nn",
    packages=find_packages(),
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
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.0.0",
        ],
        "image": [
            "pillow>=9.0.0",
        ],
        "video": [
            "opencv-python>=4.5.0",
        ],
        "audio": [
            "librosa>=0.9.0",
        ],
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
    ext_modules=[CMakeExtension("pydnn._dnn_core", sourcedir="..")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
