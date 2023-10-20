from skbuild import setup

_cmake_args = []


setup(
    name="ASCsoft",
    version="0.0.1",
    author="DHL",
    license="MIT",
    packages=["ASCsoft"],
    cmake_args=_cmake_args,
)
