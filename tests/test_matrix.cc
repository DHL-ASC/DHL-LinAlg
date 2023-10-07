#include <matrix.h>
#include <vector.h>

#include <iostream>

int main()
{
    size_t m = 5;
    size_t n = 4;
    bla::Matrix<double> A(m, n), B(m, n), C(n, m);

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
        }
    }
    bla::Matrix<double> D = A + B;
    bla::Matrix<double> E = A * B;

    std::cout << "A = " << A << std::endl;
    std::cout << "B = " << B << std::endl;
    std::cout << "A+B = " << D << std::endl;
    std::cout << "A*B = " << E << std::endl;
    std::cout << "A*x = " << A * x << std::endl;
    std::cout << "A.T = " << A.Transpose() << std::endl;
}
