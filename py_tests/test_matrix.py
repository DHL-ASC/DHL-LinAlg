import pickle

import pytest
import numpy as np

from ASCsoft.bla import Matrix
from ASCsoft.bla import Vector


@pytest.fixture(name="v3")
def vector_3():
    x = Vector(3)
    for i in range(len(x)):
        x[i] = i
    return x


@pytest.fixture(name="m3")
def matrix_3_3():
    m = Matrix(3, 3)
    for i in range(3):
        for j in range(3):
            m[i, j] = i + 2 * j
    return m


def test_matrix_vector_multiplication(m3, v3):
    res = m3 * v3
    assert res[0] == pytest.approx(10, 1e-16)
    assert res[1] == pytest.approx(13, 1e-16)
    assert res[2] == pytest.approx(16, 1e-16)


def test_matrix_set_get(m3):
    m3[1, 2] = 11
    m3[-1, 1] = 20
    assert m3[1, 2] == 11
    assert m3[2, 1] == 20


def test_matrix_slicing(m3):
    y = m3[1, :-1]
    assert isinstance(y, Vector)
    assert len(y) == 2
    col1 = m3[:, 1]
    assert len(col1) == 3
    print(m3)
    print(m3[:-1, :-1])


def test_matrix_buffer_and_transpose(m3):
    a = np.asarray(m3).T
    m3_T = m3.T()
    assert np.array_equal(a, np.asarray(m3_T))


def test_matrix_pickle(m3):
    pickel_m = pickle.dumps(m3)
    pickeld_m = pickle.loads(pickel_m)
    assert m3[1, 2] == pickeld_m[1, 2]
    assert m3[0, 1] == pickeld_m[0, 1]
