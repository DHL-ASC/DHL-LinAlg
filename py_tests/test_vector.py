import numpy as np
# search for libraray like bla.cpython-312-darwin.so in the build directory:
# import sys
# sys.path.append('/Users/joachim/texjs/lva/ws2324/ScientificComputing/ASC-bla/build')
# from bla import Vector

# import from the installed ASCsoft package:
from dhllinalg.bla import Vector


def test_vector_slicing():
    x = Vector(5)
    x[:] = 1
    assert np.array_equal(np.asarray(x), np.ones(5))
    x[1::2] = 2
    assert x[0] == 1
    assert x[3] == 2


def test_vector_add():
    x = Vector(5)
    y = Vector(5)

    for i in range(len(x)):
        x[i] = i
    y[:] = 2

    z = x + y
    assert np.array_equal(np.asarray(z), np.array([2, 3, 4, 5, 6]))


def test_vector_scal_mult():
    x = Vector(5)

    for i in range(len(x)):
        x[i] = i
    z = 2 * x
    assert np.array_equal(np.asarray(z), np.array([0, 2, 4, 6, 8]))
