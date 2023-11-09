#ifndef FILE_MATRIX_H
#define FILE_MATRIX_H

#include <iostream>
#include <memory> //for shared_ptr
#include <exception>
#include "expression.h"
#include "vector.h"
#include <simd.h>

#include <taskmanager.h>
// namespace py = pybind11;

namespace bla
{
    enum ORDERING
    {
        ColMajor,
        RowMajor
    };

    template <typename T = double, ORDERING ORD = ORDERING::RowMajor>
    class MatrixView;
    template <typename T = double, ORDERING ORD = ORDERING::RowMajor>
    class Matrix;
    template <typename T, ORDERING ORD>
    class MatrixView : public MatExpr<MatrixView<T, ORD>>
    {
    protected:
        size_t rows_, cols_, dist_;
        T *data_;

    public:
        MatrixView(size_t rows, size_t cols, size_t dist, T *data)
            : rows_(rows), cols_(cols), dist_(dist), data_(data) {}

        template <typename TB>
        MatrixView &operator=(const MatExpr<TB> &m2)
        {
            ASC_HPC::TaskManager::RunParallel([this, &m2](int id, int numThreads)
                                              {
                for (size_t i = id; i < this->rows_; i+=numThreads)
                    for (size_t j = 0; j < this->cols_; j++)
                        (*this)(i, j) = m2(i, j); });
            return *this;
        }

        MatrixView &operator=(T scal)
        {
            for (size_t i = 0; i < rows_; i++)
                for (size_t j = 0; j < cols_; j++)
                    (*this)(i, j) = scal;
            return *this;
        }
        size_t Dist() const { return dist_; }
        auto Upcast() const { return MatrixView(rows_, cols_, dist_, data_); }
        size_t nRows() const { return rows_; }
        size_t nCols() const { return cols_; }
        T *Data() { return data_; }
        T *Data() const { return data_; }
        T &operator()(size_t i) { return data_[i]; }
        const T &operator()(size_t i) const { return data_[i]; }
        T &operator()(size_t i, size_t j)
        {
            if (ORD == RowMajor)
                return data_[i * dist_ + j];
            else
                return data_[i + j * dist_];
        }
        const T &operator()(size_t i, size_t j) const
        {
            if (ORD == RowMajor)
                return data_[i * dist_ + j];
            else
                return data_[i + j * dist_];
        }

        auto Row(size_t i) const
        {
            if constexpr (ORD == RowMajor)
                return VectorView<T>(cols_, data_ + i * dist_);
            else
                return VectorView<T, size_t>(cols_, dist_, data_ + i);
        }
        auto Col(size_t i) const
        {
            if constexpr (ORD == ColMajor)
                return VectorView<T>(rows_, data_ + i * dist_);
            else
                return VectorView<T, size_t>(rows_, dist_, data_ + i);
        }

        auto Rows(size_t first, size_t next) const
        {
            if constexpr (ORD == ColMajor)
                return MatrixView(next - first, cols_, dist_, data_ + first);
            else
                return MatrixView(next - first, cols_, dist_, data_ + first * dist_);
        }

        auto Cols(size_t first, size_t next) const
        {
            if constexpr (ORD == RowMajor)
                return MatrixView(rows_, next - first, dist_, data_ + first);
            else
                return MatrixView(rows_, next - first, dist_, data_ + first * dist_);
        }

        auto Transpose()
        {
            if constexpr (ORD == RowMajor)
                return MatrixView<T, ColMajor>(nCols(), nRows(), nCols(), Data());
            else
                return MatrixView<T, RowMajor>(nCols(), nRows(), nRows(), Data());
        }

        void Pivot(size_t row, size_t *d, Matrix<T, ORD> *inv, Matrix<T, ORD> *cpy)
        {
            size_t i = row;
            for (; i < nRows(); i++)
            {
                if ((*cpy)(i, row) != 0)
                    break;
            }
            if (i == nRows())
                throw std::invalid_argument("Matrix is singular");
            if (i != row)
            {
                // TODO: implement row swapping in an efficient way
                // without moving data in memory
                // d[i] = row;
                // d[row] = i;
                for (size_t j = 0; j < nCols(); j++)
                {
                    std::swap((*inv)(i, j), (*inv)(row, j));
                    std::swap((*cpy)(i, j), (*cpy)(row, j));
                }
            }
        }

        Matrix<T, ORD> Inverse()
        {
            size_t dim = nRows();
            Matrix<T, ORD> inv(dim, dim);
            Matrix<T, ORD> cpy = (*this);
            size_t *d = new size_t[dim];

            for (size_t i = 0; i < dim; i++)
            {
                d[i] = i;
                for (size_t j = 0; j < dim; j++)
                    inv(i, j) = (i == j) ? 1 : 0;
            }

            for (size_t j = 0; j < dim; j++)
            {
                cpy.Pivot(j, d, &inv, &cpy);
                inv.Row(d[j]) = 1 / cpy(d[j], j) * inv.Row(d[j]);
                cpy.Row(d[j]) = 1 / cpy(d[j], j) * cpy.Row(d[j]);
                for (size_t i = 0; i < dim; i++)
                {
                    if (d[i] == d[j])
                        continue;
                    T s = cpy(d[i], j);
                    cpy.Row(d[i]) = -s * cpy.Row(d[j]) + cpy.Row(d[i]);
                    inv.Row(d[i]) = -s * inv.Row(d[j]) + inv.Row(d[i]);
                }
            }
            delete[] d;
            return inv;
        }
    };

    template <typename T, ORDERING ORD>
    class Matrix : public MatrixView<T, ORD>
    {
        typedef MatrixView<T, ORD> BASE;
        using BASE::cols_;
        using BASE::data_;
        using BASE::dist_;
        using BASE::rows_;

    public:
        Matrix(size_t rows, size_t cols)
            : MatrixView<T, ORD>(rows, cols, ORD == RowMajor ? cols : rows, new T[rows * cols])
        {
        }

        Matrix(const Matrix &m) : Matrix(m.nRows(), m.nCols()) { *this = m; }

        Matrix(Matrix &&m) : MatrixView<T, ORD>(0, 0, 0, nullptr)
        {
            std::swap(rows_, m.rows_);
            std::swap(cols_, m.cols_);
            std::swap(data_, m.data_);
            std::swap(dist_, m.dist_);
        }

        template <typename TB>
        Matrix(const MatExpr<TB> &m)
            : Matrix(m.nRows(), m.nCols())
        {
            *this = m;
        }

        ~Matrix() { delete[] data_; }

        using BASE::operator=;
        Matrix &operator=(const Matrix &m2)
        {
            for (size_t i = 0; i < m2.nRows() * m2.nCols(); i++)
                data_[i] = m2(i);
            return *this;
        }

        Matrix &operator=(Matrix &&m2)
        {
            for (size_t i = 0; i < m2.nRows() * m2.nCols(); i++)
                data_[i] = m2(i);
            return *this;
        }
    };

    template <typename... Args>
    std::ostream &operator<<(std::ostream &ost, const MatrixView<Args...> &m)
    {
        for (size_t i = 0; i < m.nRows(); i++)
        {
            for (size_t j = 0; j < m.nCols(); j++)
            {
                ost << m(i, j) << ", ";
            }
            ost << "\n ";
        }
        return ost;
    }

    template <typename T, ORDERING ORD>
    Matrix<T, ORD> InnerProduct(const MatrixView<T, ORD> &m1, const MatrixView<T, ORD> &m2)
    {
        Matrix<T, RowMajor> res(m1.nRows(), m2.nCols());
        // std::cout << res.nCols() << std::endl;
        ASC_HPC::TaskManager::RunParallel([&m1, &m2, &res](int id, int numThreads)
                                          {

       
        size_t i = 2*id;
        for (; res.nRows() > 1 + 2 * (numThreads - 1) && i < res.nRows() - 1 ; i += 2 * numThreads)
        {
            // std::cout << "inside for loop if" << std::endl;
            size_t j = 0;
            for (; res.nCols() > 15  && j < res.nCols() - 15 ; j += 16)
            {
                // std::cout<<"2x simd16, (i,j)=" << i << ", " << j<< ", id: " << id << std::endl;
                ASC_HPC::SIMD<double, 16> sum00(0.0);
                ASC_HPC::SIMD<double, 16> sum10(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 16> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i, k)), y1, sum00);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i + 1, k)), y1, sum10);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            }
            for (; res.nCols() > 7 && j < res.nCols() - 7 ; j += 8)
            {
                // std::cout<<"2x simd8, (i,j)=" << i << ", " << j<< ", id: " << id<< std::endl;
                ASC_HPC::SIMD<double, 8> sum00(0.0);
                ASC_HPC::SIMD<double, 8> sum10(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 8> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 8>(m1(i, k)), y1, sum00);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 8>(m1(i + 1, k)), y1, sum10);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            }
            for (; res.nCols() > 3  && j < res.nCols() - 3 ; j += 4)
            {
                // std::cout<<"2x simd4, (i,j)=" << i << ", " << j<< ", id: " << id<< std::endl;
                ASC_HPC::SIMD<double, 4> sum00(0.0);
                ASC_HPC::SIMD<double, 4> sum10(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 4> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i, k)), y1, sum00);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i + 1, k)), y1, sum10);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            }
            // not working with avx as simd1 has no store
            // for (; j < res.nCols() - 2; j += 2)
            // {
            //     ASC_HPC::SIMD<double, 2> sum00(0.0);
            //     ASC_HPC::SIMD<double, 2> sum10(0.0);
            //     for (size_t k = 0; k < m2.nRows(); k++)
            //     {
            //         ASC_HPC::SIMD<double, 2> y1(m2.Data() + k * m2.nCols() + j);
            //         sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 2>(m1(i, k)), y1, sum00);
            //         sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 2>(m1(i + 1, k)), y1, sum10);
            //     }
            //     sum00.Store(res.Data() + i * res.nCols() + j);
            //     sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            // }
            for (; j < res.nCols(); ++j)
            {
                // std::cout<<"2x simd0, (i,j)=" << i << ", " << j<< ", id: " << id<< std::endl;
                res(i, j) = 0;
                res(i + 1, j) = 0;
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    res(i, j) += m1(i, k) * m2(k, j);
                    res(i + 1, j) += m1(i + 1, k) * m2(k, j);
                }
            }
        }
        if (i+2  < res.nRows())
        {
            // std::cout << "inside first if" << std::endl;
            size_t j = 0;
            for (; res.nCols() > 15  && j < res.nCols() - 15 ; j += 16)
            {
                // std::cout<<"2x simd16, (i,j)=" << i << ", " << j<< ", id: " << id << std::endl;
                ASC_HPC::SIMD<double, 16> sum00(0.0);
                ASC_HPC::SIMD<double, 16> sum10(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 16> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i, k)), y1, sum00);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i + 1, k)), y1, sum10);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            }
            for (; res.nCols() > 7 && j < res.nCols() - 7 ; j += 8)
            {
                // std::cout<<"2x simd8, (i,j)=" << i << ", " << j<< ", id: " << id<< std::endl;
                ASC_HPC::SIMD<double, 8> sum00(0.0);
                ASC_HPC::SIMD<double, 8> sum10(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 8> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 8>(m1(i, k)), y1, sum00);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 8>(m1(i + 1, k)), y1, sum10);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            }
            for (; res.nCols() > 3  && j < res.nCols() - 3 ; j += 4)
            {
                // std::cout<<"2x simd4, (i,j)=" << i << ", " << j<< ", id: " << id<< std::endl;
                ASC_HPC::SIMD<double, 4> sum00(0.0);
                ASC_HPC::SIMD<double, 4> sum10(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 4> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i, k)), y1, sum00);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i + 1, k)), y1, sum10);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            }
            // not working with avx as simd1 has no store
            // for (; j < res.nCols() - 2; j += 2)
            // {
            //     ASC_HPC::SIMD<double, 2> sum00(0.0);
            //     ASC_HPC::SIMD<double, 2> sum10(0.0);
            //     for (size_t k = 0; k < m2.nRows(); k++)
            //     {
            //         ASC_HPC::SIMD<double, 2> y1(m2.Data() + k * m2.nCols() + j);
            //         sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 2>(m1(i, k)), y1, sum00);
            //         sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 2>(m1(i + 1, k)), y1, sum10);
            //     }
            //     sum00.Store(res.Data() + i * res.nCols() + j);
            //     sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
            // }
            for (; j < res.nCols(); ++j)
            {
                // std::cout<<"2x simd0, (i,j)=" << i << ", " << j<< ", id: " << id<< std::endl;
                res(i, j) = 0;
                res(i + 1, j) = 0;
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    res(i, j) += m1(i, k) * m2(k, j);
                    res(i + 1, j) += m1(i + 1, k) * m2(k, j);
                }
            }} });

        if (res.nRows() % 2)
        {
            // std::cout << "inside last if" << std::endl;
            size_t i = res.nRows() - 1;
            size_t id = 0;
            size_t j = 0;
            for (; res.nCols() > 15 && j < res.nCols() - 15; j += 16)
            {
                // std::cout << "simd16, (i,j)=" << i << ", " << j << ", id: " << id << std::endl;
                ASC_HPC::SIMD<double, 16> sum00(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 16> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i, k)), y1, sum00);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
            }
            for (; res.nCols() > 7 && j < res.nCols() - 7; j += 8)
            {
                // std::cout << "simd8, (i,j)=" << i << ", " << j << ", id: " << id << std::endl;
                ASC_HPC::SIMD<double, 8> sum00(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 8> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 8>(m1(i, k)), y1, sum00);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
            }
            for (; res.nCols() > 3 && j < res.nCols() - 3; j += 4)
            {
                // std::cout << "simd4, (i,j)=" << i << ", " << j << ", id: " << id << std::endl;
                ASC_HPC::SIMD<double, 4> sum00(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 4> y1(m2.Data() + k * m2.nCols() + j);
                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i, k)), y1, sum00);
                }
                sum00.Store(res.Data() + i * res.nCols() + j);
            }
            // not working with avx as simd1 has no store
            //  for (; j < res.nCols() - 2; j += 2)
            //  {
            //      ASC_HPC::SIMD<double, 2> sum00(0.0);
            //      for (size_t k = 0; k < m2.nRows(); k++)
            //      {
            //          ASC_HPC::SIMD<double, 2> y1(m2.Data() + k * m2.nCols() + j);
            //          sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 2>(m1(i, k)), y1, sum00);
            //      }
            //      sum00.Store(res.Data() + i * res.nCols() + j);
            //  }
            for (; j < res.nCols(); ++j)
            {
                // std::cout << "simd0, (i,j)=" << i << ", " << j << ", id: " << id << std::endl;
                res(i, j) = 0;
                for (size_t k = 0; k < m2.nRows(); k++)
                    res(i, j) += m1(i, k) * m2(k, j);
            }
        }
        return res;
    }

    template <size_t SIZE>
    Matrix<double, RowMajor> smallInnerProduct(const MatrixView<double, RowMajor> &m1, const MatrixView<double, RowMajor> &m2)
    {
        Matrix<double, RowMajor> res(SIZE, SIZE);
        size_t i = 0;
        for (; i < SIZE - 1; i += 2)
        {
            for (size_t j = 0; j < SIZE - 15; j += 16)
            {
                ASC_HPC::SIMD<double, 16> sum00(0.0);
                ASC_HPC::SIMD<double, 16> sum10(0.0);
                // ASC_HPC::SIMD<double, 16> sum20(0.0);
                // ASC_HPC::SIMD<double, 16> sum30(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 16> y1(m2.Data() + k * m2.nCols() + j);

                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i, k)), y1, sum00);

                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i + 1, k)), y1, sum10);

                    // sum20 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i+2, k)), y1, sum20);

                    // sum30 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 16>(m1(i+3, k)), y1, sum30);
                }

                sum00.Store(res.Data() + i * SIZE + j);

                sum10.Store(res.Data() + (i + 1) * SIZE + j);

                // sum20.Store(res.Data() + (i+2) * SIZE + j);

                // sum30.Store(res.Data() + (i+3) * SIZE + j);
            }
        }

        return res;
    }

    void (*name)(int arg1, int arg2);

    // compile inner product for small matrix sizes as template
    Matrix<double, RowMajor> (*dispatch_MatMatMult[15])(const MatrixView<double, RowMajor> &, const MatrixView<double, RowMajor> &);
    auto init_MatMatMult = []()
    {
        dispatch_MatMatMult[0] = &smallInnerProduct<16>;
        dispatch_MatMatMult[1] = &smallInnerProduct<32>;
        dispatch_MatMatMult[2] = &smallInnerProduct<48>;
        dispatch_MatMatMult[3] = &smallInnerProduct<64>;
        dispatch_MatMatMult[4] = &smallInnerProduct<80>;
        dispatch_MatMatMult[5] = &smallInnerProduct<96>;
        dispatch_MatMatMult[6] = &smallInnerProduct<112>;
        dispatch_MatMatMult[7] = &smallInnerProduct<128>;
        dispatch_MatMatMult[8] = &smallInnerProduct<144>;
        dispatch_MatMatMult[9] = &smallInnerProduct<160>;
        dispatch_MatMatMult[10] = &smallInnerProduct<176>;
        dispatch_MatMatMult[11] = &smallInnerProduct<192>;
        dispatch_MatMatMult[12] = &smallInnerProduct<208>;
        dispatch_MatMatMult[13] = &smallInnerProduct<224>;
        // Iterate<std::size(dispatch_multAB)-1> ([&] (auto i)
        // { dispatch_multAB[i] = &MultMatMat_intern; });
        dispatch_MatMatMult[14] = &InnerProduct<double, RowMajor>;
        return 1;
    }();

    Matrix<double, RowMajor> compiledInnerProduct(const MatrixView<double, RowMajor> &m1, const MatrixView<double, RowMajor> &m2)
    {
        size_t wa = m1.nCols() > 224 ? 14 : (m1.nCols() / 16 - 1);
        return (*dispatch_MatMatMult[wa])(m1, m2);
    }


    template <size_t H, size_t W>
    inline void SmallestMultMatMatKernel(MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, size_t, size_t);

    template <size_t H, size_t W>
    void MultMatMatKernel(size_t, double*, size_t, double*, size_t, double*, size_t);

//error: partial specialization not allowed
    // template <size_t W>
    // inline void SmallestMultMatMatKernel<1,W>(size_t, double, size_t, double, size_t, double, size_t, size_t);

    // template<size_t H>
    // inline void MultMatMat2<H,1>(MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, size_t);

    template <size_t H, size_t W>
    inline void SmallestMultMatMatKernel(MatrixView<double, RowMajor> A, MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C, size_t i, size_t j){
        for(;i+H<=C.nRows(); i+=H){
                MultMatMatKernel<H, W>(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
        }
        if constexpr (H!=1)
            SmallestMultMatMatKernel<H-1,W>(A,B,C,i,j);
    }

    template <size_t H, size_t W>
    void MultMatMatKernel(size_t Aw, double *Ai, size_t Adist, double *Bj, size_t Bdist, double *Cij, size_t Cdist)
    {
        ASC_HPC::SIMD<double, W> sum[H];
        for(size_t l =0;l<H;++l)
            sum[l] = ASC_HPC::SIMD<double, W>(0.0);
        for (size_t k = 0; k < Aw; ++k)
        {
            ASC_HPC::SIMD<double, W> y1(Bj + k * Bdist);
            
            for(size_t l =0;l<H;++l)
                sum[l] = ASC_HPC::FMA(ASC_HPC::SIMD<double, W>(*(Ai + k +l * Adist)), y1, sum[l]);
        }
        for(size_t l =0;l<H;++l){
            sum[l] += ASC_HPC::SIMD<double, W>(Cij + l * Cdist);
            sum[l].Store(Cij + l*Cdist);
        }
    }

    template<size_t H, size_t W>
    inline void MultMatMat2(MatrixView<double, RowMajor> A, MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C, size_t j)
    {
        for (; j + W <= C.nCols(); j += W)
        {
            size_t i = 0;
            for (; i + H <= C.nRows(); i += H){
                // if(!i){
                //     InitMultMatMatKernel<H, W>(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
                // }else{
                    MultMatMatKernel<H, W>(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
                // }
            }
            SmallestMultMatMatKernel<4, W>(A,B,C,i,j);
        }
        if constexpr (W!=1)
            MultMatMat2<H,W-1>(A, B, C, j);
    }

    void MultMatMat(const MatrixView<double, RowMajor> A, const MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C)
    {
        constexpr size_t BH = 144;
        constexpr size_t BW = 144;//168//144//96
        alignas(64) double memBA[BH * BW];
        for (size_t i1 = 0; i1 < A.nRows(); i1 += BH)
        {
            size_t j1 = 0;
            for (; j1 < A.nCols(); j1 += BW)
            {
                size_t i2 = std::min(A.nRows(), i1 + BH);
                size_t j2 = std::min(A.nCols(), j1 + BW);

                MatrixView Ablock(i2 - i1, j2 - j1, BW, memBA);
                Ablock = A.Rows(i1, i2).Cols(j1, j2);
                size_t j=0;
                MultMatMat2<4, 12>(Ablock, B.Rows(j1, j2), C.Rows(i1, i2),j);
            }
        }
    }

    template <typename T, ORDERING ORD>
    Matrix<T, ORD> InnerProduct2(const MatrixView<T, ORD> &m1, const MatrixView<T, ORD> &m2)
    {
        Matrix<T, RowMajor> res(m1.nRows(), m2.nCols());
        res = 0;
        MultMatMat(m1, m2, res);
        return res;
    }

} // namespace bla

#endif
