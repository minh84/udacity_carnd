#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "spline.h"

int add(int i, int j) {
    return i + j;
}

using namespace tk;
namespace py = pybind11;

PYBIND11_MODULE(cppext, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cppext

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");

    py::class_<spline>(m, "spline")
        .def(py::init<>())
        .def("set_points", 
            &spline::set_points, 
            py::arg("x"),
            py::arg("y"),
            py::arg("cubic_spline") = true)
        .def("__call__", [](const spline& f, double x) { return f(x); });

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
