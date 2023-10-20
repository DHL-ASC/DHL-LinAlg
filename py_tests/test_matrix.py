import pytest

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
