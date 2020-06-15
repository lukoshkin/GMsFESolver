#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "EPDE.h"
#include "GEP.h"
#include "IF.h"

using namespace dolfin;
namespace py = pybind11;

using matvec_pair = std::pair<
    std::shared_ptr<Matrix>,
    std::shared_ptr<Vector>>;
using matmat_pair = std::pair<
    std::shared_ptr<Matrix>,
    std::shared_ptr<Matrix>>;

matmat_pair
unloaded_matrices(std::shared_ptr<Function>& k) {
    auto V = k->function_space();
    EPDE::BilinearForm a(V, V);
    GEP::BilinearForm q(V, V);
    a.set_coefficient("k", k);
    q.set_coefficient("k", k);

    std::shared_ptr<Matrix> M(new Matrix);
    std::shared_ptr<Matrix> S(new Matrix);
    assemble(*M, q);
    assemble(*S, a);
    return std::make_pair(M, S);
}


matvec_pair
assemble_Ab(
        std::shared_ptr<Function>& k,
        std::shared_ptr<Function>& f) {
    auto V = k->function_space();
    EPDE::BilinearForm a(V, V);
    a.k = k;

    EPDE::LinearForm L(V);
    L.f = f;
    
    std::shared_ptr<Matrix> A(new Matrix);
    std::shared_ptr<Vector> b(new Vector);
    assemble(*A, a);
    assemble(*b, L);
    return std::make_pair(A, b);
}


matvec_pair
assemble_Ab(std::shared_ptr<Function>& k) {
    auto V = k->function_space();
    EPDE::BilinearForm a(V, V);
    a.k = k;

    auto f = std::make_shared<Constant>(0.);
    EPDE::LinearForm L(V);
    L.f = f;
    
    std::shared_ptr<Matrix> A(new Matrix);
    std::shared_ptr<Vector> b(new Vector);
    assemble(*A, a);
    assemble(*b, L);
    return std::make_pair(A, b);
}


py::array_t<double>
mass_integral_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::shared_ptr<Function>>& F1,
        std::vector<std::shared_ptr<Function>>& F2) {

    auto V = k->function_space();
    size_t N_EL(F1.size());

    IF::Form_M mass_form(V->mesh());
    mass_form.set_coefficient(0, k);

    py::array_t<double> mass({N_EL, N_EL});
    auto M_info = mass.request();
    double * M_ptr = (double *)M_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        mass_form.set_coefficient(1, F1[i]);
        for (uint j=i; j<N_EL; ++j) {
            mass_form.set_coefficient(2, F2[j]);
            M_ptr[i*N_EL+j] = assemble(mass_form);
        }
    }
    return mass;
}


py::array_t<double>
stiffness_integral_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::shared_ptr<Function>>& F1,
        std::vector<std::shared_ptr<Function>>& F2) {

    auto V = k->function_space();
    size_t N_EL(F1.size());

    IF::Form_S stiffness_form(V->mesh());
    stiffness_form.set_coefficient(0, k);

    py::array_t<double> stiffness({N_EL, N_EL});
    auto S_info = stiffness.request();
    double * S_ptr = (double *)S_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        stiffness_form.set_coefficient(1, F1[i]);
        for (uint j=i; j<N_EL; ++j) {
            stiffness_form.set_coefficient(2, F2[j]);
            S_ptr[i*N_EL+j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}


py::array_t<double>
mass_integral_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::shared_ptr<Function>>& F) 
{return mass_integral_matrix(k, F, F);}


py::array_t<double>
stiffness_integral_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::shared_ptr<Function>>& F)
{return stiffness_integral_matrix(k, F, F);}


PYBIND11_MODULE(SIGNATURE, m) {
    m.doc() = "Faster calculations with C++";
    m.def(
            "unloaded_matrices", 
            &unloaded_matrices,
            "Assemble mass and stiffness matrices",
            py::arg("kappa"));

    m.def(
            "assemble_Ab", 
            (matvec_pair (*)(
                std::shared_ptr<Function>&)
            ) &assemble_Ab,
            "Assemble system Ax=b",
            py::arg("kappa"));

    m.def(
            "assemble_Ab", 
            (matvec_pair (*)(
                std::shared_ptr<Function>&,
                std::shared_ptr<Function>&)
            ) &assemble_Ab,
            "Assemble system Ax=b",
            py::arg("kappa"), py::arg("source"));

    m.def(
            "mass_integral_matrix",
            (py::array_t<double>(*)(
                std::shared_ptr<Function>&,
                std::vector<std::shared_ptr<Function>>&,
                std::vector<std::shared_ptr<Function>>&)
            ) &mass_integral_matrix,
            "Compute generalized mass matrix <kappa Psi1_i, Psi2_j>",
            py::arg("kappa"), py::arg("Psi1"), py::arg("Psi2"));

    m.def(
            "mass_integral_matrix",
            (py::array_t<double>(*)(
                std::shared_ptr<Function>&,
                std::vector<std::shared_ptr<Function>>&)
            ) &mass_integral_matrix,
            "Assemble mass matrix <kappa Psi_i, Psi_j>",
            py::arg("kappa"), py::arg("Psi"));

    m.def(
            "stiffness_integral_matrix",
            (py::array_t<double>(*)(
                std::shared_ptr<Function>&,
                std::vector<std::shared_ptr<Function>>&,
                std::vector<std::shared_ptr<Function>>&)
            ) &stiffness_integral_matrix,
            "Compute generalized stiffness matrix <kappa grad(Psi1_i), grad(Psi2_j)>",
            py::arg("kappa"), py::arg("Psi1"), py::arg("Psi2"));

    m.def(
            "stiffness_integral_matrix",
            (py::array_t<double>(*)(
                std::shared_ptr<Function>&,
                std::vector<std::shared_ptr<Function>>&)
            ) &stiffness_integral_matrix,
            "Assemble stiffness matrix <kappa grad(Psi_i), grad(Psi_j)>",
            py::arg("kappa"), py::arg("Psi"));
}
