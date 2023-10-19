from skbuild import setup
import sys

_cmake_args = []



setup(
    name="ASCsoft",
    version="0.0.2",
    author="J. Schoeberl",
    license="MIT",
    packages=["ASCsoft"],
    cmake_args = _cmake_args
)
