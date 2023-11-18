#include <matrix.h>
#include <vector.h>
#include <chrono>

#include <iostream>
#include <taskmanager.h>

using namespace std;

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
    {
    
    int k = 200;
    bla::Matrix<double> m(k,k);
    bla::Matrix<double> n(k,k);
    for(int i=0;i<k;++i){
        for(int j=0;j<k;++j){
            m(i,j) = i+j;
            n(i,j) = 2*i+j;
        }
    }
    
    ASC_HPC::TaskManager tm(true);
    tm.StartWorkers();
    auto start = std::chrono::high_resolution_clock::now();
    auto a = bla::InnerProduct(m,n);
    auto end = std::chrono::high_resolution_clock::now();
    tm.StopWorkers();

    double time = std::chrono::duration<double, std::milli>(end-start).count();
    cout << "a(0,0) = " << a(0,0) << endl;
    cout <<" time = " << time 
           << " ms, GFlops = " << (k*k*k)/time/1e6
           << endl;
    
    }

    
    // cout << "\n\ntest InnerProduct:\n\n";
    // {
    //     int rm = 16;
    //     int cm = 9;
    //     int rn = 9;
    //     int cn = 16;
    //     bla::Matrix<double> m(rm,cm);
    //     bla::Matrix<double> n(rn,cn);

    //     for(int i=0;i<rm;++i){
    //         for(int j=0;j<cn;++j){
    //             m(i,j) = 1;
    //         }
    //     }
    //     for(int i=0;i<rn;++i){
    //         for(int j=0;j<cn;++j){
    //             n(i,j) = 1;
    //         }
    //     }

    //     cout << bla::InnerProduct(n,m) <<endl;
    // }
}
