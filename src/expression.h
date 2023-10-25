#ifndef FILE_EXPRESSION_H
#define FILE_EXPRESSION_H



namespace bla
{

    template <typename T>
    class VecExpr
    {
    public:
        auto Upcast() const { return static_cast<const T &>(*this); }
        size_t Size() const { return Upcast().Size(); }
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
        auto sum = 0.0;
        for (size_t i = 0; i < v1.Size(); i++)
            sum += v1(i) * v2(i);
        return sum;
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
            return m1_.Row(i) * m2_.Col(j);
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
