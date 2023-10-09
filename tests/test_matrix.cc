#include <matrix.h>
#include <vector.h>

#include <iostream>

int main()
{
    size_t m = 5;
    size_t n = 4;
    bla::Matrix<double> A(m, n), B(m, n), C(n, m), M(3, 3);

    bla::Vector<double> x(n);

    for (size_t i = 0; i < x.Size(); i++)
    {
        x(i) = i;
    }

    for (size_t i = 0; i < A.NumRows(); i++)
    {
        for (size_t j = 0; j < A.NumCols(); j++)
        {
            A(i, j) = i + j;
            B(i, j) = i * j;
            C(j, i) = i * j;
        }
    }

    M(0, 0) = 1;
    M(1, 0) = 2;
    M(2, 0) = -1;
    M(0, 1) = 2;
    M(1, 1) = 1;
    M(2, 1) = 2;
    M(0, 2) = -1;
    M(1, 2) = 2;
    M(2, 2) = 1;

    bla::Matrix<double> D = A + B;
    bla::Matrix<double> E = A * C;
    bla::Matrix<double, bla::RowMajor> F = A * C;

    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;
    std::cout << "C = " << C << std::endl;
    std::cout << "A+B = " << D << std::endl;
    std::cout << "A*C = " << E << std::endl;
    std::cout << "(RowMajor) A*C = " << F << std::endl;
    std::cout << "A*x = " << A * x << std::endl;
    std::cout << "A.T = " << A.Transpose() << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "M.I = " << M.I() << std::endl;
    std::cout << "M * M.I = " << M * M.I() << std::endl;
}
