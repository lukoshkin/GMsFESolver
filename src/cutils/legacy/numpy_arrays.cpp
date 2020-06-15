#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "P1.h"

using namespace dolfin;
namespace py = pybind11;


py::array_t<double>
build_mass_matrix(py::array_t<double>& F, py::array_t<double>& K) {
    auto F_info = F.request();
    auto K_info = K.request();
    double * F_ptr = (double *)F_info.ptr;
    double * K_ptr = (double *)K_info.ptr;
    size_t N_EL(F_info.shape[0]), DIM(K_info.size);

    auto mesh = std::make_shared<UnitSquareMesh>(N_EL/4, N_EL/4);
    auto V = std::make_shared<P1::Form_M_FunctionSpace_0>(mesh);

    auto u = std::make_shared<Function>(V);
    auto v = std::make_shared<Function>(V);
    auto k = std::make_shared<Function>(V);
    k->vector()->set_local(
            std::vector<double>(K_ptr, K_ptr+DIM));

    P1::Form_M mass_form(mesh, k, u, v);
    py::array_t<double> mass({N_EL, N_EL});

    auto M_info = mass.request();
    double * M_ptr = (double *)M_info.ptr;
    for (uint i=0; i<N_EL; ++i) {
        u->vector()->set_local(
                std::vector<double>(F_ptr+i*DIM, F_ptr+(i+1)*DIM));
        for (uint j=i; j<N_EL; ++j) {
            v->vector()->set_local(
                    std::vector<double>(F_ptr+j*DIM, F_ptr+(j+1)*DIM));
            M_ptr[i*N_EL+j] = assemble(mass_form);
        }
    }
    return mass;
}


py::array_t<double>
build_stiffness_matrix(py::array_t<double>& F, py::array_t<double>& K) {
    auto F_info = F.request();
    auto K_info = K.request();
    double * F_ptr = (double *)F_info.ptr;
    double * K_ptr = (double *)K_info.ptr;
    size_t N_EL(F_info.shape[0]), DIM(K_info.size);

    auto mesh = std::make_shared<UnitSquareMesh>(N_EL/4, N_EL/4);
    auto V = std::make_shared<P1::Form_S_FunctionSpace_0>(mesh);

    auto u = std::make_shared<Function>(V);
    auto v = std::make_shared<Function>(V);
    auto k = std::make_shared<Function>(V);
    k->vector()->set_local(
            std::vector<double>(K_ptr, K_ptr+DIM));

    P1::Form_S stiffness_form(mesh, k, u, v);
    py::array_t<double> stiffness({N_EL, N_EL});

    auto S_info = stiffness.request();
    double * S_ptr = (double *)S_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        u->vector()->set_local(
                std::vector<double>(F_ptr+i*DIM, F_ptr+(i+1)*DIM));
        for (uint j=i; j<N_EL; ++j) {
            v->vector()->set_local(
                    std::vector<double>(F_ptr+j*DIM, F_ptr+(j+1)*DIM));
            S_ptr[i*N_EL+j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}

PYBIND11_MODULE(SIGNATURE, m) {
    m.doc() = "A faster construction of mass and stiffness forms with C++";
    m.def(
            "build_mass_matrix",
            &build_mass_matrix, 
            "<K grad(u_i), grad(u_j)>",
            py::arg("Psi"), py::arg("kappa"));

    m.def(
            "build_stiffness_matrix",
            &build_stiffness_matrix,
            "<K u_i, u_j>",
            py::arg("Psi"), py::arg("kappa"));
}
