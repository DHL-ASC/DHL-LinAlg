#ifndef FILE_MATRIX_H
#define FILE_MATRIX_H


#define KERNEL_WIDTH 12
#define KERNEL_HEIGHT 4

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

    template <size_t H, size_t W, bool INIT = false>
    inline void SmallestMultMatMat(MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, size_t, size_t) noexcept;

    template <size_t H, size_t W, bool INIT = false>
    inline void MultMatMatKernel(size_t, double *, size_t, double *, size_t, double *, size_t) noexcept;

    template <size_t H, size_t W, bool INIT = false>
    void MultMatMat2(MatrixView<double, RowMajor>, MatrixView<double, RowMajor>, MatrixView<double, RowMajor>);

    void (*dispatch_MatMatMult[KERNEL_HEIGHT+1][KERNEL_WIDTH+1][2])(size_t, double *, size_t, double *, size_t, double *, size_t);

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

    template <size_t H, size_t W, bool INIT>
    inline void SmallestMultMatMat(MatrixView<double, RowMajor> A, MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C, size_t i, size_t j) noexcept
    {
        for (; i + H <= C.nRows(); i += H)
            MultMatMatKernel<H, W, INIT>(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());

        if constexpr (H != 1)
        {
            if (i != C.nRows())
                SmallestMultMatMat<H / 2, W, INIT>(A, B, C, i, j);
        }
    }

    template <size_t H, size_t W, bool INIT>
    inline void MultMatMatKernel(size_t Aw, double *Ai, size_t Adist, double *Bj, size_t Bdist, double *Cij, size_t Cdist) noexcept
    {   
        if constexpr (H==0 || W==0)
            return;
        else{
            ASC_HPC::SIMD<double, W> sum[H];
            for (size_t l = 0; l < H; ++l)
            {
                if constexpr (!INIT)
                    sum[l] = ASC_HPC::SIMD<double, W>(Cij + l * Cdist);
                else
                    sum[l] = ASC_HPC::SIMD<double, W>(0.0);
            }
            for (size_t k = 0; k < Aw; ++k)
            {
                ASC_HPC::SIMD<double, W> y1(Bj + k * Bdist);

                for (size_t l = 0; l < H; ++l)
                    sum[l] = ASC_HPC::FMA(ASC_HPC::SIMD<double, W>(*(Ai + k + l * Adist)), y1, sum[l]);
            }
            for (size_t l = 0; l < H; ++l)
            {
                sum[l].Store(Cij + l * Cdist);
            }
        }
    }

    template <size_t H, size_t W, bool INIT>
    void MultMatMat2(MatrixView<double, RowMajor> A, MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C)
    {
        size_t j = 0;
        size_t i;
        for (; j + W <= C.nCols(); j += W){
            for (i=0; i + H <= C.nRows(); i += H){
                (*dispatch_MatMatMult[H][W][INIT])(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
            }
            (*dispatch_MatMatMult[C.nRows()-i][W][INIT])(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
        }
        for (i=0; i + H <= C.nRows(); i += H){
            (*dispatch_MatMatMult[H][C.nCols()-j][INIT])(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
        }
        (*dispatch_MatMatMult[C.nRows()-i][C.nCols()-j][INIT])(A.nCols(), &A(i, 0), A.Dist(), &B(0, j), B.Dist(), &C(i, j), C.Dist());
    }

    void MultMatMat(const MatrixView<double, RowMajor> A, const MatrixView<double, RowMajor> B, MatrixView<double, RowMajor> C)
    {
        ASC_HPC::TaskManager::RunParallel([&A, &B, &C](int id, int numThreads)
                                          {     
        size_t numThreads = 1;
        size_t id =0;      

        constexpr size_t BH = 96;
        constexpr size_t BW = 96; // 168//144//96
        size_t i1 = id * BH;
        alignas(64) double memBA[BH * BW];
        for (; i1 < A.nRows(); i1 += BH * numThreads)
        {
            // std::cout << "id: "<< id << "i1: " << i1 << std::endl;
            size_t j1 = 0;
            for (; j1 < A.nCols(); j1 += BW)
            {
                size_t i2 = std::min(A.nRows(), i1 + BH);
                size_t j2 = std::min(A.nCols(), j1 + BW);

                MatrixView Ablock(i2 - i1, j2 - j1, BW, memBA);
                Ablock = A.Rows(i1, i2).Cols(j1, j2);
                if (!j1)
                    MultMatMat2<4, 12, true>(Ablock, B.Rows(j1, j2), C.Rows(i1, i2));
                else
                    MultMatMat2<4, 12>(Ablock, B.Rows(j1, j2), C.Rows(i1, i2));
            }
        } });
    }

    template <typename T, ORDERING ORD>
    Matrix<T, ORD> InnerProduct(const MatrixView<T, ORD> &m1, const MatrixView<T, ORD> &m2)
    {
        Matrix<T, RowMajor> res(m1.nRows(), m2.nCols());
        MultMatMat(m1, m2, res);
        return res;
    }


    template<size_t I, size_t J>
    void InnerIteratorDispatch(){
        dispatch_MatMatMult[I][J][0] = &MultMatMatKernel<I,J>;
        dispatch_MatMatMult[I][J][1] = &MultMatMatKernel<I,J,true>;
        
        if constexpr (J!=0)
            InnerIteratorDispatch<I,J-1>();
    }

    template<size_t I, size_t J>
    void IteratorDispatch(){
        InnerIteratorDispatch<I,J>();
        
        if constexpr (I!=0)
            IteratorDispatch<I-1,J>();
    }


    auto init_MatMatMult = []()
    {
        IteratorDispatch<KERNEL_HEIGHT,KERNEL_WIDTH> ();

        return 1;
    }();

} // namespace bla

#endif
