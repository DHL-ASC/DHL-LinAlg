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

    for (size_t i = 0; i < A.nRows(); i++)
    {
        for (size_t j = 0; j < A.nCols(); j++)
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

    std::cout << "A = " << A << std::endl;
    // Transpose
    std::cout << "A.Transpose() = " << A.Transpose() << std::endl;
    // Mat vec product
    std::cout << "A*x = " << A * x << std::endl;
    // Rows and cols of matrix
    std::cout << "A.Row(1) = " << A.Row(1) << std::endl;
    std::cout << "2*A.Row(1) = " << 2 * A.Row(1) << std::endl;
    std::cout << "A.Rows(1, 3) = " << A.Rows(1, 3) << std::endl;
    std::cout << "A.Cols(1, 3) = " << A.Cols(1, 3) << std::endl;
    std::cout << "A.Rows(1,3).Cols(1, 2) = " << A.Rows(1, 3).Cols(1, 2) << std::endl;
    // Set row
    A.Row(1) = 1 * A.Row(0);
    std::cout << "A.Row(1) = " << A.Row(1) << std::endl;
    // mat mat addition
    std::cout << "B = " << B << std::endl;
    std::cout << "A + B = " << A + B << std::endl;
    // mat mat multiplication
    std::cout << "C = " << C << std::endl;
    std::cout << "A * C = " << A * C << std::endl;
    // Inverse
    std::cout << "M = " << M << std::endl;
    auto minv = M.Inverse();
    std::cout << "M.Inverse() = " << minv << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "M * M.Inverse() = " << M * minv << std::endl;




    {
        bla::Vector<double> x(3);
        bla::Vector<double> res(3);
        bla::Matrix<double> m(3,3);
        for(size_t i=0; i< x.Size(); ++i)
            x(i) = i;

        for(size_t i=0; i< m.nRows(); ++i)
            for(size_t j=0; j< m.nCols(); ++j)
                m(i, j) = i + 2 * j;

        res = m * x;

        std::cout << "M*x = " << res << std::endl;
    }


}
