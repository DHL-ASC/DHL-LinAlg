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
        DHL_HPC::ParallelComputingTF t;

    public:
        ParallelComputing() : t() {}
        ParallelComputing(size_t nthreads) : t(nthreads) {}
        ParallelComputing(bool trace) : t(trace) {}
        ParallelComputing(size_t nthreads, bool trace) : t(nthreads, trace) {}
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
            return DHL_HPC::ParallelComputingTF::getNumThreads();
        }
    };
}

PYBIND11_MODULE(bla, m)
{
    m.doc() = "Basic linear algebra module"; // optional module docstring
    m.def("NumThreads", &ParallelComputing::getNumThreads, "Get number of threads in use");

    m.def(
        "InnerProduct",
        [](Matrix<double, RowMajor> &self, Matrix<double, RowMajor> &other)
        {
            return InnerProduct(self, other);
        },
        py::arg("u"),
        py::arg("v"),"Matrix multiplication of u and v.");
    py::class_<ParallelComputing>(m, "ParallelComputing")
        .def(py::init<>(),"Set number of threads to hardware concurrency.")
        .def(py::init<size_t>(), py::arg("nThreads"), "Set number of threads to \"nThreads\".")
        .def(py::init<bool>(), py::arg("Trace"), "Set number of threads to hardware concurrency and prepare to write trace file.")
        .def(py::init<size_t, bool>(), py::arg("nThreads"), py::arg("Trace"), "Set number of threads to \"nThreads\" and prepare to write trace file.")
        .def("__enter__", &ParallelComputing::Enter, "Start parallel computing (multithreading).")
        .def("__exit__", &ParallelComputing::Exit, "Stop parallel computing (multithreading).");
    //.def("__timing__", &DHL_HPC::TaskManager::Timing);

    py::class_<Vec<3, double>>(m, "Vec3D")
        .def(py::init<>(),
             "Create vector of length \"size\"")
        .def("__len__", &Vec<3,double>::Size, "Return length of vector")
        .def("__setitem__", [](Vec<3, double> &self, int i, double v)
             {
        if (i < 0)
            i += self.Size();
        if (i < 0 || i >= self.Size())
            throw py::index_error("vector index out of range");
        self(i) = v; },
             py::arg("index"),
             py::arg("value"),"Set value of item with index \"index\"")
        .def("__getitem__", [](Vec<3, double> &self, int i)
             {
        if (i < 0)
            i += self.Size();
        if (i < 0 || i >= self.Size())
            throw py::index_error("vector index out of range");
        return self(i); },
             py::arg("index"),"Get value of item with index \"index\"")
        .def("__str__", [](const Vec<3, double> &self)
             {
        std::stringstream str;
        str << self;
        return str.str(); }, "Print vector elements.");

    py::class_<Vector<double>>(m, "Vector")
        .def(py::init<size_t>(),
             py::arg("size"), "Create vector of length \"size\"")
        .def("__len__", &Vector<double>::Size, "Return length of vector")

        .def("__setitem__", [](Vector<double> &self, int i, double v)
             {
        if (i < 0)
            i += self.Size();
        if (i < 0 || i >= self.Size())
            throw py::index_error("vector index out of range");
        self(i) = v; },
             py::arg("index"),
             py::arg("value"),"Set value of item with index \"index\"")
        .def("__getitem__", [](Vector<double> &self, int i)
             {
        if (i < 0)
            i += self.Size();
        if (i < 0 || i >= self.Size())
            throw py::index_error("vector index out of range");
        return self(i); },
             py::arg("index"),"Get value of item with index \"index\"")
        .def("__setitem__", [](Vector<double> &self, py::slice inds, double val)
             {
        size_t start, stop, step, n;
        InitSlice(inds, self.Size(), start, stop, step, n);
        self.Range(start, stop).Slice(0, step) = val; },
             py::arg("indices"),
             py::arg("value"),"Set value of items with index in \"indices\"")

        .def("__add__", [](Vector<double> &self, Vector<double> &other)
             { return Vector<double>(self + other); },
             py::arg("other"),"Vector addition")

        .def("__rmul__", [](Vector<double> &self, double scal)
             { return Vector<double>(scal * self); },
             py::arg("scal"),"Vector scaling")

        .def("__mul__", [](Vector<double> &self, Vector<double> &other)
             { return self * other; },
             py::arg("other"),"Vector product")

        .def("__str__", [](const Vector<double> &self)
             {
        std::stringstream str;
        str << self;
        return str.str(); }, "Print vector elements.")

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
            }),"Pickle bytestream");

    py::class_<Matrix<double, RowMajor>>(m, "Matrix", py::buffer_protocol())
        .def(py::init<size_t, size_t>(),
             py::arg("rows"), py::arg("cols"),
             "Create matrix of \"rows\" rows and \"cols\" columns.")
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
             },
             py::arg("index"),"Get value of item with index \"index\"")
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
             },
             py::arg("indices"),"Get values of items with index in \"indices\"")
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
             },
             py::arg("indices"),"Get values of items with index in \"indices\"")
        // slice over rows and cols
        .def("__getitem__",
             [](Matrix<double, RowMajor> self, std::tuple<py::slice, py::slice> ind)
             {
                 auto [row_slice, col_slice] = ind;
                 size_t row_start, row_stop, row_step, row_n;
                 size_t col_start, col_stop, col_step, col_n;
                 InitSlice(row_slice, self.nRows(), row_start, row_stop, row_step, row_n);
                 InitSlice(col_slice, self.nCols(), col_start, col_stop, col_step, col_n);
                 return Matrix<double, RowMajor>(self.Rows(row_start, row_stop).Cols(col_start, col_stop));
             },
             py::arg("indices"),"Get values of items with index in \"indices\"")
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
        self(i, j) = val; },
             py::arg("index"),
             py::arg("value"),"Set value of item with index \"index\" to \"value\"")
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
             },
             py::arg("indices"),
             py::arg("value"),"Set value of items with index \"indices\" to \"value\"")
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
             },
             py::arg("indices"),
             py::arg("value"),"Set value of items with index \"indices\" to \"value\"")
        // set value on rows and cols
        .def("__setitem__",
             [](Matrix<double, RowMajor> &self, std::tuple<py::slice, py::slice> ind, double val)
             {
                 auto [row_slice, col_slice] = ind;
                 size_t row_start, row_stop, row_step, row_n;
                 size_t col_start, col_stop, col_step, col_n;
                 InitSlice(row_slice, self.nRows(), row_start, row_stop, row_step, row_n);
                 InitSlice(col_slice, self.nCols(), col_start, col_stop, col_step, col_n);
                 self.Rows(row_start, row_stop).Cols(col_start, col_stop) = val;
             },
             py::arg("indices"),
             py::arg("value"),"Set value of items with index \"indices\" to \"value\"")
        .def("__add__",
             [](Matrix<double, RowMajor> &self, Matrix<double, RowMajor> &other)
             {
                 return Matrix<double, RowMajor>(self + other);
             },
             py::arg("other"),"Matrix addition")
        // matrix scalar multiplication
        .def("__mul__",
             [](Matrix<double, RowMajor> &self, double scal)
             {
                 return Matrix<double, RowMajor>(scal * self);
             },
             py::arg("scal"),"Matrix scaling")
        .def("__rmul__",
             [](Matrix<double, RowMajor> &self, double scal)
             {
                 return Matrix<double, RowMajor>(scal * self);
             },
             py::arg("scal"),"Matrix scaling")

        // matrix matrix multiplication
        .def("__mul__",
             [](Matrix<double, RowMajor> &self, Matrix<double, RowMajor> &other)
             {
                 return Matrix<double, RowMajor>(self * other);
             },
             py::arg("other"),"Matrix multipliaction (deprecated, use InnerProduct instead)")

        // matrix vector multiplication
        .def("__mul__",
             [](Matrix<double, RowMajor> &self, Vector<double> &other)
             {
                 return Vector<double>(self * other);
             },
             py::arg("vec"),"Matrix vector multipliaction")
        .def("__rmul__",
             [](Matrix<double, RowMajor> &self, Vector<double> &other)
             {
                 return Vector<double>(self * other);
             },
             py::arg("vec"),"Matrix vector multipliaction")

        .def("__str__",
             [](const Matrix<double, RowMajor> &self)
             {
                 std::stringstream str;
                 str << self;
                 return str.str();
             },"Print matrix elements.")
        .def(
            "I", [](Matrix<double, RowMajor> &self)
            { return self.Inverse(); },
            "Matrix inverse.")
        .def(
            "T", [](Matrix<double, RowMajor> &self)
            { return Matrix<double, RowMajor>(self.Transpose()); },
            "Matrix transpose.")
        .def_property_readonly(
            "shape",
            [](const Matrix<double, RowMajor> &self)
            { return std::tuple(self.nRows(), self.nCols()); }, "Get matrix shape as tuple[rows, cols].")
        .def_property_readonly(
            "nrows", [](const Matrix<double, RowMajor> &self)
            { return self.nRows(); },
            "Get number of rows.")
        .def_property_readonly(
            "ncols", [](const Matrix<double, RowMajor> &self)
            { return self.nCols(); },
            "Get number of columns.")
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
            }),"Pickle bytestream");
}
