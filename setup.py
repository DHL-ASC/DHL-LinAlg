from skbuild import setup

_cmake_args = []

test_deps = ["pytest>=7.4.2", "numpy>=1.26.1"]
extras = {
    "test": test_deps,
}

setup(
    name="ASCsoft",
    version="0.0.1",
    author="DHL",
    license="MIT",
    packages=["ASCsoft"],
    tests_require=test_deps,
    extras_require=extras,
    cmake_args=_cmake_args,
)
