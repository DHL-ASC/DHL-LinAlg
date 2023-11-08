import pickle

import pytest
import numpy as np

from dhllinalg.bla import Matrix, Vector, InnerProduct


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


def matrix_matrix_multiplication(matrix_sizes, method):
    matrix_size1 = matrix_sizes[0]
    matrix_size2 = matrix_sizes[1]

    m = Matrix(matrix_size1[0], matrix_size1[1])
    n = Matrix(matrix_size2[0], matrix_size2[1])
    for i in range(matrix_size1[0]):
        for j in range(matrix_size1[1]):
            m[i, j] = i + 2 * j

    for i in range(matrix_size2[0]):
        for j in range(matrix_size2[1]):
            n[i, j] = 2 * i + 3 * j

    numpy_m = np.asarray(m)
    numpy_n = np.asarray(n)

    c = Matrix(matrix_size1[0], matrix_size2[1])
    d = Matrix(matrix_size2[0], matrix_size1[1])
    if method == "normal":
        c = m * n
        d = n * m
    elif method == "InnerProduct":
        c = InnerProduct(m, n)
        d = InnerProduct(n, m)

    assert np.array_equal(np.asarray(c), np.dot(numpy_m, numpy_n))
    assert np.array_equal(np.asarray(d), np.dot(numpy_n, numpy_m))


matrix_sizes = [
    ((10, 5), (5, 10)),  # Test with 10x5 and 5x10 matrices
    ((66, 34), (34, 66)),  # Test with 6x3 and 3x6 matrices
    ((47, 13), (13, 47)),  # Test with 47x13 and 13x47 matrices
    ((128, 7), (7, 128)),  # Test with 128x7 and 7x128 matrices
    ((9, 83), (83, 9)),  # Test with 9x83 and 83x9 matrices
    ((512, 1024), (1024, 512)),  # Test with 9x83 and 83x9 matrices
]
methods = ["normal", "InnerProduct"]
matrix_size_and_methods = [
    (size, method) for size in matrix_sizes for method in methods
]


@pytest.mark.parametrize("matrix_sizes, method", matrix_size_and_methods)
def test_matrix_multiplication_with_various_sizes(matrix_sizes, method):
    matrix_matrix_multiplication(matrix_sizes, method)


def test_matrix_set_get(m3):
    m3[1, 2] = 11
    m3[-1, 1] = 20
    assert m3[1, 2] == 11
    assert m3[2, 1] == 20


def test_matrix_inverse():
    m = Matrix(3, 3)
    m[0, 0] = 0
    m[1, 0] = 2
    m[2, 0] = -1
    m[0, 1] = 2
    m[1, 1] = 1
    m[2, 1] = 2
    m[0, 2] = -1
    m[1, 2] = 2
    m[2, 2] = 1
    minv = m.I()
    assert np.allclose(np.asarray(minv), np.linalg.inv(m))
    assert np.array_equal(np.asarray(m * minv), np.eye(3))
    assert np.array_equal(np.asarray(minv * m), np.eye(3))


def test_matrix_slicing(m3):
    y = m3[1, :-1]
    col1 = m3[:, 1]
    sub_matrix = m3[:-1, :-1]
    assert len(col1) == 3
    assert isinstance(y, Vector)
    assert len(y) == 2
    assert np.array_equal(sub_matrix, np.array([[0, 2], [1, 3]]))
    n = Matrix(10, 5)
    n[:, :] = 1
    assert np.array_equal(np.asarray(n), np.ones((10, 5)))


def test_matrix_buffer_and_transpose(m3):
    a = np.asarray(m3).T
    m3_T = m3.T()
    assert np.array_equal(a, np.asarray(m3_T))


def test_matrix_pickle(m3):
    pickel_m = pickle.dumps(m3)
    pickeld_m = pickle.loads(pickel_m)
    assert m3[1, 2] == pickeld_m[1, 2]
    assert m3[0, 1] == pickeld_m[0, 1]
