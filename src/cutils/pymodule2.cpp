#include <dolfin.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "Projection.h"
#include "EPDE.h"
#include "GEP.h"
#include "GC.h"
#include "MIF.h"


using namespace dolfin;
namespace py = pybind11;

//using matvec_pair = std::pair<
//    std::shared_ptr<Matrix>,
//    std::shared_ptr<Vector>>;
//using matmat_pair = std::pair<
//    std::shared_ptr<Matrix>,
//    std::shared_ptr<Matrix>>;
//
using matmat_pair = std::pair<Matrix, Matrix>;
using matvec_pair = std::pair<Matrix, Vector>;


matmat_pair
unloaded_matrices(std::shared_ptr<Function>& k) {
    auto V = k->function_space();
    auto comm = V->mesh()->mpi_comm();
    EPDE::BilinearForm a(V, V);
    GEP::BilinearForm q(V, V);
    a.set_coefficient("k", k);
    q.set_coefficient("k", k);

    //std::shared_ptr<Matrix> M(new Matrix);
    //std::shared_ptr<Matrix> S(new Matrix);
    //assemble(*M, q);
    //assemble(*S, a);

    Matrix M(comm), S(comm);
    assemble(M, q);
    assemble(S, a);
    return std::make_pair(M, S);
}


matvec_pair
assemble_Ab(
        std::shared_ptr<Function>& k,
        std::shared_ptr<Function>& f) {
    auto V = k->function_space();
    auto comm = V->mesh()->mpi_comm();
    EPDE::BilinearForm a(V, V);
    EPDE::LinearForm L(V);
    a.k = k;
    L.f = f;
    
    //std::shared_ptr<Matrix> A(new Matrix);
    //std::shared_ptr<Vector> b(new Vector);
    //assemble(*A, a);
    //assemble(*b, L);
    Matrix A(comm); Vector b(comm);
    assemble(A, a);
    assemble(b, L);
    return std::make_pair(A, b);
}


matvec_pair
assemble_Ab(std::shared_ptr<Function>& k) {
    auto V = k->function_space();
    auto comm = V->mesh()->mpi_comm();
    auto f = std::make_shared<Constant>(0.);
    EPDE::BilinearForm a(V, V);
    EPDE::LinearForm L(V);
    a.k = k;
    L.f = f;
    
    //std::shared_ptr<Matrix> A(new Matrix);
    //std::shared_ptr<Vector> b(new Vector);
    //assemble(*A, a);
    //assemble(*b, L);
    Matrix A(comm); Vector b(comm);
    assemble(A, a);
    assemble(b, L);
    return std::make_pair(A, b);
}


matvec_pair
diagonal_coupling(
        std::shared_ptr<Function>& k,
        std::shared_ptr<Function>& xi,
        std::shared_ptr<Function>& f) {
    auto V = k->function_space();
    auto comm = V->mesh()->mpi_comm();
    GC::BilinearForm a(V, V);
    a.xi_1 = xi;
    a.xi_2 = xi;
    a.k = k;

    GC::LinearForm L(V);
    L.xi_2 = xi;
    L.f = f;
    
    //std::shared_ptr<Matrix> A(new Matrix);
    //std::shared_ptr<Vector> b(new Vector);
    //assemble(*A, a);
    //assemble(*b, L);

    Matrix A(comm); Vector b(comm);
    assemble(A, a);
    assemble(b, L);
    return std::make_pair(A, b);
}


Matrix
offdiagonal_coupling(
        std::shared_ptr<Function>& k,
        std::shared_ptr<Function>& xi_1,
        std::shared_ptr<Function>& xi_2) {
    auto V = k->function_space();
    auto comm = V->mesh()->mpi_comm();
    GC::BilinearForm a(V, V);
    a.xi_1 = xi_1;
    a.xi_2 = xi_2;
    a.k = k;
    
    //std::shared_ptr<Matrix> A(new Matrix);
    //assemble(*A, a);
    Matrix A(comm);
    assemble(A, a);
    return A;
}


std::vector<std::vector<double>>
multiply_project(
        std::shared_ptr<Function>& xi,
        py::array_t<double>& Nv) {
    auto V = xi->function_space();
    auto comm = V->mesh()->mpi_comm();
    Projection::BilinearForm a(V, V);
    Projection::LinearForm L(V);
    L.xi = xi;

    auto Nv_info = Nv.request();
    double * Nv_ptr = (double *)Nv_info.ptr;
    size_t N_EL(Nv_info.shape[0]), DIM(V->dim());

    Matrix A(comm); Vector b(comm);
    assemble(A, a);

    auto f = std::make_shared<Function>(V);
    auto u = std::make_shared<Function>(V);
    std::vector<std::vector<double>> Nv_ms(
            N_EL, std::vector<double>(DIM));

    for (uint i=0; i<N_EL; ++i) {
        f->vector()->set_local(
                std::vector<double>(
                    Nv_ptr+i*DIM, Nv_ptr+i*DIM+DIM));
        L.f = f;
        assemble(b, L);
        solve(A, *u->vector(), b);
        u->vector()->get_local(Nv_ms[i]);
    }
   return Nv_ms;
}


std::vector<std::vector<double>>
zero_extrapolation(
        std::vector<std::vector<double>>& Nv,
        std::shared_ptr<FunctionSpace>& V,
        std::shared_ptr<FunctionSpace>& W) {

    size_t N_EL(Nv.size());
    auto v = std::make_shared<Function>(V);
    auto e = std::make_shared<Function>(W);
    std::vector<std::vector<double>> ms_dofs(
            N_EL, std::vector<double>(W->dim()));

    for (uint i=0; i<N_EL; ++i) {
        v->vector()->set_local(Nv[i]);
        LagrangeInterpolator::interpolate(*e, *v);
        e->vector()->get_local(ms_dofs[i]);
    }
    return ms_dofs;
}


std::vector<std::vector<double>>
zero_extrapolation(
        py::array_t<double>& Nv, 
        std::shared_ptr<FunctionSpace>& V,
        std::shared_ptr<FunctionSpace>& W) {

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
        LagrangeInterpolator::interpolate(*e, *v);
        e->vector()->get_local(ms_dofs[i]);
    }
    return ms_dofs;
}


std::vector<std::shared_ptr<Function>>
compose(
        py::array_t<double>& ms_dofs,
        std::shared_ptr<FunctionSpace>& W) {

    auto ms_info = ms_dofs.request();
    double * ms_ptr = (double *)ms_info.ptr;
    size_t N1(ms_info.shape[0]), N2(ms_info.shape[1]), DIM(W->dim());

    std::vector<std::shared_ptr<Function>> basis(N1*N2);
    for (uint i=0; i<N1; ++i) {
        for (uint j=0; j<N2; ++j) {
            auto w = std::make_shared<Function>(W);
            w->vector()->set_local(
                    std::vector<double>(
                        ms_ptr + i*N2*DIM + j*DIM,
                        ms_ptr + i*N2*DIM + j*DIM + DIM));
            basis[i*N2+j] = w;
        }
    }
    return basis;
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
        for (uint j=0; j<N_EL; ++j) {
            v->vector()->set_local(Psi2[j]);
            stiffness_form.set_coefficient(2, v);
            S_ptr[i*N_EL+j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}


std::pair<py::array_t<double>, py::array_t<double>>
integral_assembling(
        std::shared_ptr<Function>& k,
        std::vector<std::vector<double>>& Nv,
        std::shared_ptr<Function>& rhs, 
        std::shared_ptr<MeshFunction<size_t>>& markers) {

    auto W = k->function_space();
    auto u = std::make_shared<Function>(W);
    auto v = std::make_shared<Function>(W);
    size_t N_EL(Nv.size());

    MIF::Form_S stiffness_form(W->mesh());
    stiffness_form.set_coefficient(0, k);
    stiffness_form.dx = markers;

    MIF::Form_F src_term(W->mesh());
    src_term.set_coefficient(0, rhs);
    src_term.dx = markers;

    py::array_t<double> A({N_EL, N_EL});
    auto A_info = A.request();
    double * A_ptr = (double *)A_info.ptr;

    py::array_t<double> b(N_EL);
    auto b_info = b.request();
    double * b_ptr = (double *)b_info.ptr;

    for (uint i=0; i<N_EL; ++i) {
        u->vector()->set_local(Nv[i]);
        stiffness_form.set_coefficient(1, u);
        src_term.set_coefficient(1, u);
        b_ptr[i] = assemble(src_term);
        for (uint j=i; j<N_EL; ++j) {
            v->vector()->set_local(Nv[j]);
            stiffness_form.set_coefficient(2, v);
            A_ptr[i*N_EL+j] = assemble(stiffness_form);
        }
    }
    return std::make_pair(A, b);
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
            "multiply_project",
            &multiply_project,
            "Multiply by xi and project to xi->function_space()",
            py::arg("xi"), py::arg("NodalValues"));

    m.def(
            "zero_extrapolation",
            (std::vector<std::vector<double>>(*)(
                std::vector<std::vector<double>>&,
                std::shared_ptr<FunctionSpace>&,
                std::shared_ptr<FunctionSpace>&)
            ) &zero_extrapolation,
            "Extrapolates local ms functions to W with zeros",
            py::arg("NodalValues"), py::arg("V"), py::arg("W"));

    m.def(
            "zero_extrapolation",
            (std::vector<std::vector<double>>(*)(
                py::array_t<double>&,
                std::shared_ptr<FunctionSpace>&,
                std::shared_ptr<FunctionSpace>&)
            ) &zero_extrapolation,
            "Extrapolates local ms functions to W with zeros",
            py::arg("NodalValues"), py::arg("V"), py::arg("W"));

    m.def(
            "compose",
            &compose,
            "Restore function basis by their dofs",
            py::arg("ms_dofs"), py::arg("W"));

//     m.def(
//             "stiffness_integral_matrix",
//             (py::array_t<double> (*)(
//                 std::shared_ptr<Function>&,
//                 std::vector<std::vector<double>>&,
//                 std::vector<std::vector<double>>&,
//                 std::shared_ptr<MeshFunction<size_t>>)
//             ) &stiffness_integral_matrix,
//             "Assemble stiffness matrix <kappa grad(Psi_i), grad(Psi_j)>",
//             py::arg("kappa"), py::arg("Psi_i"),
//             py::arg("Psi_j"), py::arg("markers"));

    m.def(
            "stiffness_integral_matrix",
            &stiffness_integral_matrix,
            "Assemble stiffness matrix <kappa grad(Psi_i), grad(Psi_j)>",
            py::arg("kappa"), py::arg("Psi_i"),
            py::arg("Psi_j"), py::arg("markers"));

//     m.def(
//             "stiffness_integral_matrix",
//             (std::pair<py::array_t<double>, py::array_t<double>> (*)(
//                 std::shared_ptr<Function>&,
//                 std::vector<std::vector<double>>&,
//                 std::shared_ptr<Function>&,
//                 std::shared_ptr<MeshFunction<size_t>>)
//             ) &stiffness_integral_matrix,
//             "Returns A,b of system Ax = b",
//             py::arg("kappa"), py::arg("Psi"),
//             py::arg("RHS"), py::arg("markers"));

    m.def(
            "integral_assembling",
            &integral_assembling,
            "Returns A,b of system Ax = b",
            py::arg("kappa"), py::arg("Psi"),
            py::arg("RHS"), py::arg("markers"));
}
