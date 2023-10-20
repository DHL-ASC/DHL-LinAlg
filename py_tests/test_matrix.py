import pytest

from ASCsoft.bla import Matrix
from ASCsoft.bla import Vector


def test_matrix_vector_multiplication():
    x = Vector(3)
    m = Matrix(3, 3)

    for i in range(len(x)):
        x[i] = i
    for i, j in zip(range(3), range(3)):
        m[i, j] = i + 2 * j
    res = m * x
    assert res[0] == pytest.approx(0, 1e-16)
    assert res[1] == pytest.approx(3, 1e-16)
    assert res[2] == pytest.approx(12, 1e-16)
