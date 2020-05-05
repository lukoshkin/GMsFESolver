#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "EPDE.h"
#include "GEP.h"
#include "GC.h"
#include "MIF.h"


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


matvec_pair
diagonal_coupling(
        std::shared_ptr<Function>& k,
        std::shared_ptr<Function>& xi,
        std::shared_ptr<Function>& f) {
    auto V = k->function_space();
    GC::BilinearForm a(V, V);
    a.xi_1 = xi;
    a.xi_2 = xi;
    a.k = k;

    GC::LinearForm L(V);
    L.xi_1 = xi;
    L.f = f;
    
    std::shared_ptr<Matrix> A(new Matrix);
    std::shared_ptr<Vector> b(new Vector);
    assemble(*A, a);
    assemble(*b, L);
    return std::make_pair(A, b);
}


std::shared_ptr<Matrix>
offdiagonal_coupling(
        std::shared_ptr<Function>& k,
        std::shared_ptr<Function>& xi_1,
        std::shared_ptr<Function>& xi_2) {
    auto V = k->function_space();
    GC::BilinearForm a(V, V);
    a.xi_1 = xi_1;
    a.xi_2 = xi_2;
    a.k = k;
    
    std::shared_ptr<Matrix> A(new Matrix);
    assemble(*A, a);
    return A;
}

std::vector<std::vector<double>>
project_embed(
        py::array_t<double>& Nv,
        std::shared_ptr<Function>& xi,
        std::shared_ptr<FunctionSpace>& W) {

    auto V = xi->function_space();
    auto v = std::make_shared<Function>(V);
    auto e = std::make_shared<Function>(W);

    auto Nv_info = Nv.request();
    double * Nv_ptr = (double *)Nv_info.ptr;
    size_t N_EL(Nv_info.shape[0]), DIM(V->dim());

    std::vector<std::vector<double>> ms_dofs(
            N_EL, std::vector<double>(W->dim()));

    for (uint i=0; i<N_EL; ++i) {
        v->vector()->set_local(
                std::vector<double>(
                    Nv_ptr+i*DIM, Nv_ptr+i*DIM+DIM));

        *v->vector() *= *xi->vector();
        LagrangeInterpolator::interpolate(*e, *v);
        e->vector()->get_local(ms_dofs[i]);
    }
    return ms_dofs;
}


py::array_t<double>
stiffness_integral_matrix(
        std::shared_ptr<Function>& k,
        std::vector<std::vector<double>>& Psi1,
        std::vector<std::vector<double>>& Psi2,
        std::shared_ptr<MeshFunction<size_t>>& markers) {

    auto W = k->function_space();
    auto u = std::make_shared<Function>(W);
    auto v = std::make_shared<Function>(W);
    size_t N_EL(Psi1.size());

    MIF::Form_S stiffness_form(W->mesh());
    stiffness_form.set_coefficient(0, k);
    stiffness_form.dx = markers;

    py::array_t<double> stiffness({N_EL, N_EL});
    auto S_info = stiffness.request();
    double * S_ptr = (double *)S_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        u->vector()->set_local(Psi1[i]);
        stiffness_form.set_coefficient(1, u);
        for (uint j=i; j<N_EL; ++j) {
            v->vector()->set_local(Psi2[j]);
            stiffness_form.set_coefficient(2, v);
            S_ptr[i*N_EL+j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}

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
            "diagonal_coupling",
            &diagonal_coupling,
            "Computes unloaded diagonal blocks",
            py::arg("kappa"), py::arg("xi"), py::arg("RHS"));

    m.def(
            "offdiagonal_coupling",
            &offdiagonal_coupling,
            "Computes unloaded offdiagonal blocks",
            py::arg("kappa"), py::arg("xi_1"), py::arg("xi_2"));

    m.def(
            "project_embed",
            &project_embed,
            "Extrapolates local ms functions to W with zeros",
            py::arg("NodalValues"), py::arg("xi"), py::arg("W"));

    m.def(
            "stiffness_integral_matrix",
            &stiffness_integral_matrix,
            "Assemble stiffness matrix <kappa grad(Psi_i), grad(Psi_j)>",
            py::arg("kappa"), py::arg("Psi_i"),
            py::arg("Psi_j"), py::arg("markers"));
}
