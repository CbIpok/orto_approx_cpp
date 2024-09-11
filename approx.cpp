#include "approx_orto.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Для поддержки контейнеров C++ STL (vector)
// Обёртка для Pybind11
//PYBIND11_MODULE(approx_orto, m) {
//    m.def("approximate_with_non_orthogonal_basis_orto", &approximate_with_non_orthogonal_basis_orto,
//        "Approximate vector using non-orthogonal basis and return coefficients",
//        pybind11::arg("vector"), pybind11::arg("basis"));
//}


PYBIND11_MODULE(approx_orto, m) {
    m.doc() = "Module for non-orthogonal basis approximation";

    // Обёртка для функции approximate_with_non_orthogonal_basis_orto_t
    m.def("approximate_with_non_orthogonal_basis_orto", &approximate_with_non_orthogonal_basis_orto_std,
        "Approximate a vector using a non-orthogonal basis",
        pybind11::arg("vector"), pybind11::arg("basis"));
}