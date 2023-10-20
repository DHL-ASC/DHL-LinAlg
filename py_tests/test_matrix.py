import pickle

import pytest
import numpy as np

from ASCsoft.bla import Matrix
from ASCsoft.bla import Vector


def test_matrix_vector_multiplication():
    x = Vector(3)
    m = Matrix(3, 3)

    for i in range(len(x)):
        x[i] = i
    for i in range(len(x)):
        for j in range(len(x)):
            m[i, j] = i + 2 * j
    res = m * x

    assert res[0] == pytest.approx(10, 1e-16)
    assert res[1] == pytest.approx(13, 1e-16)
    assert res[2] == pytest.approx(16, 1e-16)


def test_matrix_buffer():
    m = Matrix(3, 3)
    for i in range(3):
        for j in range(3):
            m[i, j] = i + 2 * j
    a = np.asarray(m)


def test_matrix_pickle():
    m = Matrix(3, 3)
    for i in range(3):
        for j in range(3):
            m[i, j] = i + 2 * j
    pickel_m = pickle.dumps(m)
    pickeld_m = pickle.loads(pickel_m)
    assert m[1, 2] == pickeld_m[1, 2]
    assert m[0, 1] == pickeld_m[0, 1]
