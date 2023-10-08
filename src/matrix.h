#ifndef FILE_Matrix_H
#define FILE_Matrix_H

#include <iostream>
#include <memory> //for shared_ptr

#include "shape.h"
#include "vector.h"

namespace bla
{

    enum ORDERING
    {
        ColMajor,
        RowMajor
    };
    template <typename T, ORDERING ORD = ORDERING::ColMajor>
    class Matrix
    {
        size_t rows_, cols_;
        std::shared_ptr<T[]> data_;
        bool isTransposed_ = false;

    public:
        Matrix(size_t rows, size_t cols)
            : rows_(rows), cols_(cols), data_(new T[rows * cols])
        {
        }

        Matrix(size_t rows, size_t cols, std::shared_ptr<T[]> data, bool isTransposed)
            : rows_(rows), cols_(cols), data_(data)
        {
            isTransposed_ = isTransposed?false:true;
        }

        Matrix(Shape shape) : Matrix(shape.NumRows(), shape.NumCols()) { ; }

        Matrix(const Matrix &m) : Matrix(m.NumRows(), m.NumCols()) { *this = m; }

        Matrix(Matrix &&m) : rows_{0}, cols_{0}, data_(nullptr)
        {
            std::swap(rows_, m.rows_);
            std::swap(cols_, m.cols_);
            std::swap(data_, m.data_);
        }

        ~Matrix() { ; }

        Matrix &operator=(const Matrix &v2)
        {
            for (size_t i = 0; i < DataSize(); i++)
                data_[i] = v2(i);
            return *this;
        }

        Matrix &operator=(Matrix &&m2)
        {
            for (size_t i = 0; i < DataSize(); i++)
                data_[i] = m2(i);
            return *this;
        }

        Matrix<T, ORD> Transpose()
        {
            Matrix<T, ORD> trans(NumRows(), NumCols(), Data(), IsTransposed());

            return trans;
        }

        std::shared_ptr<T[]> Data() { return data_; }
        std::shared_ptr<const T[]> Data() const { return data_; }
        bool IsTransposed() const { return isTransposed_; }
        size_t NumRows() const { return rows_; }
        size_t NumCols() const { return cols_; }
        size_t DataSize() const { return rows_ * cols_; }
        const Shape &Shape() const { return Shape(NumRows(), NumCols()); }
        T &operator()(size_t i) { return data_[i]; }
        const T &operator()(size_t i) const { return data_[i]; }
        T &operator()(size_t i, size_t j)
        {
            if (ORD == RowMajor)
            {
                return IsTransposed()?Data()[j * NumCols() + i]:Data()[i * NumCols() + j];
            }
            else
            {
                return IsTransposed()?Data()[j + i * NumRows()]:Data()[i + j * NumRows()];
            }
        }
        const T &operator()(size_t i, size_t j) const
        {
            if (ORD == RowMajor)
            {
                return IsTransposed()?Data()[j * NumCols() + i]:Data()[i * NumCols() + j];
            }
            else
            {
                return IsTransposed()?Data()[j + i * NumRows()]:Data()[i + j * NumRows()];
            }
        }
    };

    template <typename T, ORDERING ORDA, ORDERING ORDB>
    Matrix<T, ORDA> operator+(const Matrix<T, ORDA> &a, const Matrix<T, ORDB> &b)
    {
        Matrix<T, ORDA> sum(a.NumRows(), a.NumCols());
        for (size_t i = 0; i < a.NumRows(); i++)
            for (size_t j = 0; j < a.NumCols(); j++)
                sum(i, j) = a(i, j) + b(i, j);
        return sum; // sum is stored as ORDA
    }

    template <typename T, ORDERING ORDA, ORDERING ORDB>
    Matrix<T, ORDA> operator*(const Matrix<T, ORDA> &a, const Matrix<T, ORDB> &b)
    {
        Matrix<T, ORDA> res(a.NumRows(), b.NumCols());
        for (size_t i = 0; i < res.NumRows(); i++)
        {
            for (size_t j = 0; j < res.NumCols(); j++)
            {
                T sum = 0;
                for (size_t k = 0l; k < res.NumCols(); k++)
                {
                    sum += a(i, k) * b(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res; // result is stored as ORDA
    }

    template <typename T, ORDERING ORD>
    Vector<T> operator*(const Matrix<T, ORD> &a, const Vector<T> &b)
    {
        Vector<T> res(a.NumRows());
        for (size_t i = 0; i < a.NumRows(); i++)
        {
            T sum = 0;
            for (size_t j = 0; j < a.NumCols(); j++)
            {
                sum += a(i, j) * b(j);
            }
            res(i) = sum;
        }
        return res;
    }

    template <typename T, ORDERING ORD>
    Vector<T> operator*(const Vector<T> &b, const Matrix<T, ORD> &a)
    {
        return a * b;
    }

    template <typename T, ORDERING ORD>
    std::ostream &operator<<(std::ostream &ost, const Matrix<T, ORD> &m)
    {
        size_t r = m.IsTransposed()?m.NumCols():m.NumRows();
        size_t c = m.IsTransposed()?m.NumRows():m.NumCols();
        for (size_t i = 0; i < r; i++)
        {
            for (size_t j = 0; j < c; j++)
            {
                ost << m(i, j) << ", ";
            }
            ost << "\n ";
        }
        return ost;
    }

} // namespace bla

#endif
