#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "P1.h"

#define N_EL 32
#define DIM (N_EL+1)*(N_EL+1)

using namespace dolfin;
namespace py = pybind11;

py::array_t<double>
build_mass_matrix(py::array_t<double>& data) {
    auto mesh = std::make_shared<UnitSquareMesh>(N_EL, N_EL);
    auto V = std::make_shared<P1::Form_M_FunctionSpace_0>(mesh);

    auto u = std::make_shared<Function>(V);
    auto v = std::make_shared<Function>(V);
    auto k = std::make_shared<Function>(V);
    auto in_info = data.request();
    double * i_ptr = (double *)in_info.ptr;

    k->vector()->set_local(
            std::vector<double>(i_ptr+4*N_EL*DIM, i_ptr+4*N_EL*DIM+DIM));

    P1::Form_M mass_form(mesh, k, u, v);
    py::array_t<double> mass(16*N_EL*N_EL);

    auto out_info = mass.request();
    double * o_ptr = (double *)out_info.ptr;

    std::vector<double> buffer(DIM);
    for (int i=0; i<4*N_EL; ++i) {
        u->vector()->set_local(
                std::vector<double>(i_ptr+i*DIM, i_ptr+(i+1)*DIM));
        for (int j=i; j<4*N_EL; ++j) {
            v->vector()->set_local(
                    std::vector<double>(i_ptr+j*DIM, i_ptr+(j+1)*DIM));
            o_ptr[i*4*N_EL+j] = assemble(mass_form);
        }
    }
    return mass;
}


py::array_t<double>
build_stiffness_matrix(py::array_t<double>& data) {
    auto mesh = std::make_shared<UnitSquareMesh>(N_EL, N_EL);
    auto V = std::make_shared<P1::Form_S_FunctionSpace_0>(mesh);

    auto u = std::make_shared<Function>(V);
    auto v = std::make_shared<Function>(V);
    auto k = std::make_shared<Function>(V);

    auto in_info = data.request();
    double * i_ptr = (double *)in_info.ptr;

    k->vector()->set_local(
            std::vector<double>(i_ptr+4*N_EL*DIM, i_ptr+4*N_EL*DIM+DIM));

    P1::Form_S stiffness_form(mesh, k, u, v);
    py::array_t<double> stiffness(16*N_EL*N_EL);

    auto out_info = stiffness.request();
    double * o_ptr = (double *)out_info.ptr;

    for (int i=0; i<4*N_EL; ++i) {
        u->vector()->set_local(
                std::vector<double>(i_ptr+i*DIM, i_ptr+(i+1)*DIM));
        for (int j=i; j<4*N_EL; ++j) {
            v->vector()->set_local(
                    std::vector<double>(i_ptr+j*DIM, i_ptr+(j+1)*DIM));
            o_ptr[i*4*N_EL+j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}

PYBIND11_MODULE(SIGNATURE, m) {
    m.def("build_mass_matrix", &build_mass_matrix);
    m.def("build_stiffness_matrix", &build_stiffness_matrix);
}
