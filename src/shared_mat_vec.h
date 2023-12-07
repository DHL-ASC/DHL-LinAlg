#ifndef FILE_SHARED_MAT_VEC_H
#define FILE_SHARED_MAT_VEC_H

#include <iostream>

namespace bla
{
     enum ORDERING
    {
        ColMajor,
        RowMajor
    };

    template <typename T = double, typename TDIST = std::integral_constant<size_t, 1>>
	class VectorView;

    template <typename T = double, ORDERING ORD = ORDERING::RowMajor>
    class MatrixView;
}

#endif