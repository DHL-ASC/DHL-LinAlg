from ASCsoft.bla import Matrix
from ASCsoft.bla import Vector


def test_matrix_vector_multiplication():
    x = Vector(3)

    for i in range(len(x)):
        x[i] = i
    m = Matrix(3, 3)
    for i, j in zip(range(3), range(3)):
        m[i, j] = i + 2 * j
    res = m * x
    assert res[0] == 0
    assert res[1] == 3
    assert res[2] == 12
