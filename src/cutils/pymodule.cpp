#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "P1.h"

using namespace dolfin;
namespace py = pybind11;


py::array_t<double>
build_mass_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::shared_ptr<Function>>& F) {

    auto V = k->function_space();
    size_t N_EL(F.size());

    P1::Form_M mass_form(V->mesh());
    mass_form.set_coefficient(0, k);

    py::array_t<double> mass({N_EL, N_EL});
    auto M_info = mass.request();
    double * M_ptr = (double *)M_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        mass_form.set_coefficient(1, F[i]);
        for (uint j=i; j<N_EL; ++j) {
            mass_form.set_coefficient(2, F[j]);
            M_ptr[i*N_EL+j] = assemble(mass_form);
        }
    }
    return mass;
}


py::array_t<double>
build_stiffness_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::shared_ptr<Function>>& F) {

    auto V = k->function_space();
    size_t N_EL(F.size());

    P1::Form_S stiffness_form(V->mesh());
    stiffness_form.set_coefficient(0, k);

    py::array_t<double> stiffness({N_EL, N_EL});
    auto S_info = stiffness.request();
    double * S_ptr = (double *)S_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        stiffness_form.set_coefficient(1, F[i]);
        for (uint j=i; j<N_EL; ++j) {
            stiffness_form.set_coefficient(2, F[j]);
            S_ptr[i*N_EL+j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}


PYBIND11_MODULE(SIGNATURE, m) {
    m.doc() = "Faster calculations with C++";
    m.def(
            "build_mass_matrix",
            &build_mass_matrix, 
            "Computes mass matrix <kappa Psi_i, Psi_j>",
            py::arg("kappa"), py::arg("Psi"));

    m.def(
            "build_stiffness_matrix",
            &build_stiffness_matrix,
            "Computes stiffness matrix <kappa grad(Psi_i), grad(Psi_j)>",
            py::arg("kappa"), py::arg("Psi"));
}
