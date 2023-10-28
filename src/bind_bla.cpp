#include <sstream>
#include <pybind11/pybind11.h>

#include "vector.h"
#include "matrix.h"
#include "taskmanager.h"

using namespace bla;
namespace py = pybind11;

// from ngsolve
void InitSlice(const py::slice &inds, size_t len, size_t &start, size_t &stop, size_t &step, size_t &n)
{
    if (!inds.compute(len, &start, &stop, &step, &n))
        throw py::error_already_set();
}

namespace bla
{
    class ParallelComputing
    {
        ASC_HPC::TaskManager t;

    public:
        ParallelComputing() : t() {}
        void Enter()
        {
            t.StartWorkers();
        }
        void Exit(py::object exc_type, py::object exc_value, py::object traceback)
        {
            t.StopWorkers();
        }
        static int getNumThreads()
        {
            return ASC_HPC::TaskManager::getNumThreads();
        }
    };
}

PYBIND11_MODULE(bla, m)
{
    m.doc() = "Basic linear algebra module"; // optional module docstring
    m.def("NumThreads", &ParallelComputing::getNumThreads, "Get number of threads used");

    py::class_<ParallelComputing>(m, "ParallelComputing")
        .def(py::init<>())
        .def("__enter__", &ParallelComputing::Enter)
        .def("__exit__", &ParallelComputing::Exit);
    //.def("__timing__", &ASC_HPC::TaskManager::Timing);
    //.def(py::init<size_t>(), "pajetrace"_a, "Run paje-tracer, specify buffersize in bytes")

    py::class_<Vector<double>>(m, "Vector")
        .def(py::init<size_t>(),
             py::arg("size"), "Create vector of given size")
        .def("__len__", &Vector<double>::Size, "Return size of vector")

        .def("__setitem__", [](Vector<double> &self, int i, double v)
             {
        if (i < 0)
            i += self.Size();
        if (i < 0 || i >= self.Size())
            throw py::index_error("vector index out of range");
        self(i) = v; })
        .def("__getitem__", [](Vector<double> &self, int i)
             {
        if (i < 0)
            i += self.Size();
        if (i < 0 || i >= self.Size())
            throw py::index_error("vector index out of range");
        return self(i); })
        .def("__setitem__", [](Vector<double> &self, py::slice inds, double val)
             {
        size_t start, stop, step, n;
        InitSlice(inds, self.Size(), start, stop, step, n);
        self.Range(start, stop).Slice(0, step) = val; })

        .def("__add__", [](Vector<double> &self, Vector<double> &other)
             { return Vector<double>(self + other); })

        .def("__rmul__", [](Vector<double> &self, double scal)
             { return Vector<double>(scal * self); })

        .def("__str__", [](const Vector<double> &self)
             {
        std::stringstream str;
        str << self;
        return str.str(); })

        .def(py::pickle(
            [](Vector<double> &self) { // __getstate__
                /* return a tuple that fully encodes the state of the object */
                return py::make_tuple(self.Size(),
                                      py::bytes((char *)(void *)&self(0), self.Size() * sizeof(double)));
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                    throw std::runtime_error("should be a 2-tuple!");

                Vector<double> v(t[0].cast<size_t>());
                py::bytes mem = t[1].cast<py::bytes>();
                std::memcpy(&v(0), PYBIND11_BYTES_AS_STRING(mem.ptr()), v.Size() * sizeof(double));
                return v;
            }));

    py::class_<Matrix<double, RowMajor>>(m, "Matrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>(),
             py::arg("rows"), py::arg("cols"),
             "Create a matrix of given size")
        // getter
        .def("__getitem__",
             [](Matrix<double, RowMajor> self, std::tuple<int, int> ind)
             {
                 auto [i, j] = ind;
                 if (i < 0)
                     i += self.nRows();
                 if (j < 0)
                     j += self.nCols();
                 if (i < 0 || i >= self.nRows())
                     throw py::index_error("matrix row out of range");
                 if (j < 0 || j >= self.nCols())
                     throw py::index_error("matrix col out of range");
                 return self(i, j);
             })
        // get row, slice over cols
        .def("__getitem__",
             [](Matrix<double, RowMajor> self, std::tuple<int, py::slice> ind)
             {
                 auto [row, slice] = ind;
                 if (row < 0)
                     row += self.nRows();
                 if (row < 0 || row >= self.nRows())
                     throw py::index_error("matrix row out of range");
                 size_t start, stop, step, n;
                 InitSlice(slice, self.nCols(), start, stop, step, n);
                 return Vector<double>(self.Row(row).Range(start, stop).Slice(0, step));
             })
        // get col, slice over rows
        .def("__getitem__",
             [](Matrix<double, RowMajor> self, std::tuple<py::slice, int> ind)
             {
                 auto [slice, col] = ind;
                 if (col < 0)
                     col += self.nCols();
                 if (col < 0 || col >= self.nCols())
                     throw py::index_error("matrix col out of range");
                 size_t start, stop, step, n;
                 InitSlice(slice, self.nRows(), start, stop, step, n);
                 return Vector<double>(self.Col(col).Range(start, stop).Slice(0, step));
             })
        // slice over rows and cols
        .def("__getitem__",
             [](Matrix<double, RowMajor> self, std::tuple<py::slice, py::slice> ind)
             {
                 auto [row_slice, col_slice] = ind;
                 size_t row_start, row_stop, row_step, row_n;
                 size_t col_start, col_stop, col_step, col_n;
                 InitSlice(row_slice, self.nRows(), row_start, row_stop, row_step, row_n);
                 InitSlice(col_slice, self.nRows(), col_start, col_stop, col_step, col_n);
                 return Matrix<double, RowMajor>(self.Rows(row_start, row_stop).Cols(col_start, col_stop));
             })
        // setter
        .def("__setitem__",
             [](Matrix<double, RowMajor> &self, std::tuple<int, int> ind,
                double val)
             {
        auto [i, j] = ind;
        if (i < 0)
            i += self.nRows();
        if (j < 0)
            j += self.nCols();
        if (i < 0 || i >= self.nRows())
            throw py::index_error("matrix row out of range");
        if (j < 0 || j >= self.nCols())
            throw py::index_error("matrix col out of range");
        self(i, j) = val; })
        // set value on row, slice over cols
        .def("__setitem__",
             [](Matrix<double, RowMajor> &self, std::tuple<int, py::slice> ind, double val)
             {
                 auto [row, slice] = ind;
                 if (row < 0)
                     row += self.nRows();
                 if (row < 0 || row >= self.nRows())
                     throw py::index_error("matrix row out of range");
                 size_t start, stop, step, n;
                 InitSlice(slice, self.nCols(), start, stop, step, n);
                 self.Rows(row, row + 1).Cols(start, stop) = val;
             })
        // set value on col
        .def("__setitem__",
             [](Matrix<double, RowMajor> &self, std::tuple<py::slice, int> ind, double val)
             {
                 auto [slice, col] = ind;
                 if (col < 0)
                     col += self.nCols();
                 if (col < 0 || col >= self.nCols())
                     throw py::index_error("matrix col out of range");
                 size_t start, stop, step, n;
                 InitSlice(slice, self.nRows(), start, stop, step, n);
                 self.Cols(col, col + 1).Rows(start, stop) = val;
             })
        // set value on rows and cols
        .def("__setitem__",
             [](Matrix<double, RowMajor> &self, std::tuple<py::slice, py::slice> ind, double val)
             {
                 auto [row_slice, col_slice] = ind;
                 size_t row_start, row_stop, row_step, row_n;
                 size_t col_start, col_stop, col_step, col_n;
                 InitSlice(row_slice, self.nRows(), row_start, row_stop, row_step, row_n);
                 InitSlice(col_slice, self.nRows(), col_start, col_stop, col_step, col_n);
                 self.Rows(row_start, row_stop).Cols(col_start, col_stop) = val;
             })
        .def("__add__",
             [](Matrix<double, RowMajor> &self, Matrix<double, RowMajor> &other)
             {
                 return Matrix<double, RowMajor>(self + other);
             })
        // matrix scalar multiplication
        .def("__mul__",
             [](Matrix<double, RowMajor> &self, double scal)
             {
                 return Matrix<double, RowMajor>(scal * self);
             })
        .def("__rmul__",
             [](Matrix<double, RowMajor> &self, double scal)
             {
                 return Matrix<double, RowMajor>(scal * self);
             })

        // matrix matrix multiplication
        .def("__mul__",
             [](Matrix<double, RowMajor> &self, Matrix<double, RowMajor> &other)
             {
                 return Matrix<double, RowMajor>(self * other);
             })
        .def("__rmul__",
             [](Matrix<double, RowMajor> &self, Matrix<double, RowMajor> &other)
             {
                 return Matrix<double, RowMajor>(self * other);
             })

        // matrix vector multiplication
        .def("__mul__",
             [](Matrix<double, RowMajor> &self, Vector<double> &other)
             {
                 return Vector<double>(self * other);
             })
        .def("__rmul__",
             [](Matrix<double, RowMajor> &self, Vector<double> &other)
             {
                 return Vector<double>(self * other);
             })

        .def("__str__",
             [](const Matrix<double, RowMajor> &self)
             {
                 std::stringstream str;
                 str << self;
                 return str.str();
             })
        .def(
            "I", [](Matrix<double, RowMajor> &self)
            { return self.Inverse(); },
            "Return the inverse of the matrix")
        .def(
            "T", [](Matrix<double, RowMajor> &self)
            { return Matrix<double, RowMajor>(self.Transpose()); },
            "Return the transpose of the matrix")
        .def_property_readonly(
            "shape",
            [](const Matrix<double, RowMajor> &self)
            { return std::tuple(self.nRows(), self.nCols()); },
            "Get matrix shape as tuple[rows, cols]")
        .def_property_readonly(
            "nrows", [](const Matrix<double, RowMajor> &self)
            { return self.nRows(); },
            "Return the number of rows of the matrix")
        .def_property_readonly(
            "ncols", [](const Matrix<double, RowMajor> &self)
            { return self.nCols(); },
            "Return the number of cols of the matrix")
        .def_buffer([](Matrix<double, RowMajor> &m) -> py::buffer_info
                    { return py::buffer_info(
                          m.Data(),                                /* Pointer to buffer */
                          sizeof(double),                          /* Size of one scalar */
                          py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                          2,                                       /* Number of dimensions */
                          {m.nRows(), m.nCols()},                  /* Buffer dimensions */
                          {sizeof(double) * m.nCols(),             /* Strides (in bytes) for each index */
                           sizeof(double)}); })
        .def(py::pickle(
            [](Matrix<double, RowMajor> &self)
            {
                return py::make_tuple(self.nRows(), self.nCols(),
                                      py::bytes((char *)(void *)&self(0), self.nRows() * self.nCols() * sizeof(double)));
            },
            [](py::tuple t)
            {
                if (t.size() != 3)
                    throw std::runtime_error("should be a 3-tuple!");

                Matrix<double, RowMajor> m(t[0].cast<size_t>(), t[1].cast<size_t>());
                py::bytes mem = t[2].cast<py::bytes>();
                std::memcpy(&m(0), PYBIND11_BYTES_AS_STRING(mem.ptr()), m.nRows() * m.nCols() * sizeof(double));
                return m;
            }));
}
