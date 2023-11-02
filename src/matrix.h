#ifndef FILE_MATRIX_H
#define FILE_MATRIX_H

#include <iostream>
#include <memory> //for shared_ptr
#include <exception>
#include "expression.h"
#include "vector.h"
#include <simd.h>

#include <taskmanager.h>

namespace bla
{
    enum ORDERING
    {
        ColMajor,
        RowMajor
    };

    template <typename T, ORDERING ORD>
    class MatrixView;
    template <typename T, ORDERING ORD = ORDERING::RowMajor>
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

    template <typename T>
    Matrix<T, RowMajor> InnerProduct(const MatrixView<T, RowMajor> &m1, const MatrixView<T, RowMajor> &m2)
    {
        Matrix<T, RowMajor> res(m1.nRows(), m2.nCols());
        size_t i = 0;
        size_t rows_res = res.nRows() / 2;
        size_t cols_res = res.nCols() / 8;

        std::cout << rows_res << std::endl;
        std::cout << cols_res << std::endl;
        for (; i < res.nRows(); i += 2)
        {
            for (size_t j = 0; j < res.nCols(); j += 8)
            {
                ASC_HPC::SIMD<double, 4> sum00(0.0);
                ASC_HPC::SIMD<double, 4> sum01(0.0);
                ASC_HPC::SIMD<double, 4> sum10(0.0);
                ASC_HPC::SIMD<double, 4> sum11(0.0);
                for (size_t k = 0; k < m2.nRows(); k++)
                {
                    ASC_HPC::SIMD<double, 4> y1(m2.Data() + k * m2.nCols() + j);
                    ASC_HPC::SIMD<double, 4> y2(m2.Data() + k * m2.nCols() + j + 4);

                    sum00 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i, k)), y1, sum00);
                    sum01 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i, k)), y2, sum01);
                    sum10 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i + 1, k)), y1, sum10);
                    sum11 = ASC_HPC::FMA(ASC_HPC::SIMD<double, 4>(m1(i + 1, k)), y2, sum11);
                }

                sum00.Store(res.Data() + i * res.nCols() + j);
                sum01.Store(res.Data() + i * res.nCols() + j + 4);
                sum10.Store(res.Data() + (i + 1) * res.nCols() + j);
                sum11.Store(res.Data() + (i + 1) * res.nCols() + j + 4);
            }
        }

        return res;
    }

} // namespace bla

#endif
