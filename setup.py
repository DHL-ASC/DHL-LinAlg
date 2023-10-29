from skbuild import setup

_cmake_args = []

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
    version="0.0.1",
    author="DHL",
    license="MIT",
    packages=["dhllinalg"],
    tests_require=test_deps,
    extras_require=extras,
    cmake_args=_cmake_args,
)
