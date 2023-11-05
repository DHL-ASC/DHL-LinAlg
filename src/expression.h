#ifndef FILE_EXPRESSION_H
#define FILE_EXPRESSION_H

#include <simd.h>

namespace bla
{

    template <typename T>
    class VecExpr
    {
    public:
        auto Upcast() const { return static_cast<const T &>(*this); }
        size_t Size() const { return Upcast().Size(); }
        // const T *Data() const { return Upcast().Data(); }
        auto Data() const { return Upcast().Data(); }
        auto operator()(size_t i) const { return Upcast()(i); }
    };

    template <typename TA, typename TB>
    class SumVecExpr : public VecExpr<SumVecExpr<TA, TB>>
    {
        TA a_;
        TB b_;

    public:
        SumVecExpr(TA a, TB b) : a_(a), b_(b) {}
        auto operator()(size_t i) const { return a_(i) + b_(i); }
        size_t Size() const { return a_.Size(); }
    };

    template <typename TA, typename TB>
    auto operator+(const VecExpr<TA> &a, const VecExpr<TB> &b)
    {
        return SumVecExpr(a.Upcast(), b.Upcast());
    }

    template <typename TSCAL, typename TV>
    class ScaleVecExpr : public VecExpr<ScaleVecExpr<TSCAL, TV>>
    {
        TSCAL scal_;
        TV vec_;

    public:
        ScaleVecExpr(TSCAL scal, TV vec) : scal_(scal), vec_(vec) {}

        auto operator()(size_t i) const { return scal_ * vec_(i); }
        size_t Size() const { return vec_.Size(); }
    };

    template <typename T>
    auto operator*(double scal, const VecExpr<T> &v)
    {
        return ScaleVecExpr(scal, v.Upcast());
    }

    template <typename TA, typename TB>
    auto operator*(const VecExpr<TA> &v1, const VecExpr<TB> &v2)
    {
        size_t i = 0;
        double r = 0;

        ASC_HPC::SIMD<double, 16> res16(0.0);
        for (; v1.Size() > 15 && i < v1.Size() - 15; i += 16)
        {
            ASC_HPC::SIMD<double, 16> s1(v1.Data() + i);
            ASC_HPC::SIMD<double, 16> s2(v2.Data() + i);
            res16 = ASC_HPC::FMA(s1, s2, res16);
        }
        r += ASC_HPC::HSum(res16);

        ASC_HPC::SIMD<double, 8> res8(0.0);
        for (; v1.Size() > 7 && i < v1.Size() - 7; i += 8)
        {
            ASC_HPC::SIMD<double, 8> s1(v1.Data() + i);
            ASC_HPC::SIMD<double, 8> s2(v2.Data() + i);
            res8 = ASC_HPC::FMA(s1, s2, res8);
        }
        r += ASC_HPC::HSum(res8);

        ASC_HPC::SIMD<double, 4> res4(0.0);
        for (; v1.Size() > 3 && i < v1.Size() - 3; i += 4)
        {
            ASC_HPC::SIMD<double, 4> s1(v1.Data() + i);
            ASC_HPC::SIMD<double, 4> s2(v2.Data() + i);
            res4 = ASC_HPC::FMA(s1, s2, res4);
        }
        r += ASC_HPC::HSum(res4);

        for (; i < v1.Size(); i++)
            r += v1(i) * v2(i);

        return r;
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &ost, const VecExpr<T> &v)
    {
        if (v.Size() > 0)
            ost << v(0);
        for (size_t i = 1; i < v.Size(); i++)
            ost << ", " << v(i);
        return ost;
    }

    //
    // Matrix
    //

    template <typename T>
    class MatExpr
    {
    public:
        auto Upcast() const { return static_cast<const T &>(*this); }
        size_t nRows() const { return Upcast().nRows(); }
        size_t nCols() const { return Upcast().nCols(); }
        auto operator()(size_t i, size_t j) const { return Upcast()(i, j); }
    };

    template <typename TA, typename TB>
    class SumMatExpr : public MatExpr<SumMatExpr<TA, TB>>
    {
        TA a_;
        TB b_;

    public:
        SumMatExpr(TA a, TB b) : a_(a), b_(b) {}
        auto operator()(size_t i, size_t j) const { return a_(i, j) + b_(i, j); }
        size_t nRows() const { return a_.nRows(); }
        size_t nCols() const { return a_.nCols(); }
    };

    template <typename TA, typename TB>
    auto operator+(const MatExpr<TA> &a, const MatExpr<TB> &b)
    {
        return SumMatExpr(a.Upcast(), b.Upcast());
    }

    template <typename TSCAL, typename TM>
    class ScaleMatExpr : public MatExpr<ScaleMatExpr<TSCAL, TM>>
    {
        TSCAL scal_;
        TM mat_;

    public:
        ScaleMatExpr(TSCAL scal, TM mat) : scal_(scal), mat_(mat) {}
        auto operator()(size_t i, size_t j) const { return scal_ * mat_(i, j); }
        size_t nRows() const { return mat_.nRows(); }
        size_t nCols() const { return mat_.nCols(); }
    };

    template <typename T>
    auto operator*(double scal, const MatExpr<T> &m)
    {
        return ScaleMatExpr(scal, m.Upcast());
    }

    template <typename TM, typename TV>
    class MatVecExpr : public VecExpr<MatVecExpr<TM, TV>>
    {
        TM m_;
        TV v_;

    public:
        MatVecExpr(TM m, TV v) : m_(m), v_(v) {}
        auto operator()(size_t i) const
        {
            return m_.Row(i) * v_;
        }
        size_t Size() const { return m_.nRows(); }
    };

    template <typename TM, typename TV>
    auto operator*(const MatExpr<TM> &m, const VecExpr<TV> &v)
    {
        return MatVecExpr(m.Upcast(), v.Upcast());
    }

    template <typename TA, typename TB>
    class MatMatExpr : public MatExpr<MatMatExpr<TA, TB>>
    {
        TA m1_;
        TB m2_;

    public:
        MatMatExpr(TA m1, TB m2) : m1_(m1), m2_(m2) {}
        auto operator()(size_t i, size_t j) const
        {
            double r0 = 0;
            double r1 = 0;
            double r2 = 0;
            double r3 = 0;
            size_t k = 0;
            for (; k < m2_.nRows() - 3; k += 4)
            {
                r0 += m1_(i, k) * m2_(k, j);
                r1 += m1_(i, k + 1) * m2_(k + 1, j);
                r2 += m1_(i, k + 2) * m2_(k + 2, j);
                r3 += m1_(i, k + 3) * m2_(k + 3, j);
            }
            for (; k < m2_.nRows(); k++)
                r0 += m1_(i, k) * m2_(k, j);

            return r0 + r1 + r2 + r3;
            // return m1_.Row(i) * m2_.Col(j);
        }
        size_t nRows() const { return m1_.nRows(); }
        size_t nCols() const { return m2_.nCols(); }
    };

    template <typename TA, typename TB>
    auto operator*(const MatExpr<TA> &m1, const MatExpr<TB> &m2)
    {
        return MatMatExpr(m1.Upcast(), m2.Upcast());
    }

    template <typename T>
    std::ostream &operator<<(std::ostream &ost, const MatExpr<T> &m)
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

}

#endif
