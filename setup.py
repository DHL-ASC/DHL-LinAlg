from skbuild import setup
import sys

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

_cmake_args = ["-DCMAKE_BUILD_TYPE=Release"]

if('win32' in sys.platform):
    _cmake_args += ['/arch:AVX2']



test_deps = ["pytest", "numpy"]
docs = ["sphinx", "myst-nb", "pandocfilters"]

all_deps = test_deps + docs

extras = {
    "test": test_deps,
    "docs": docs,
    "all": all_deps,
}

setup(
    name="dhllinalg",
    version="0.1.1",
    author="DHL-Team",
    license="MIT",
    packages=["dhllinalg"],
    description="Basic Linear Algebra in C++ (TU Vienna - ASC)",
    long_description=long_description,  # Long description read from the the readme file
    long_description_content_type="text/markdown",
    install_requires=["setuptools>=42", "scikit-build>=0.13", "pybind11"],
    tests_require=test_deps,
    extras_require=extras,
    cmake_args=_cmake_args,
    url="https://github.com/DHL-ASC/DHL-LinAlg",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
