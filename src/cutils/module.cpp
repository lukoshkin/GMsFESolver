#include <dolfin.h>
#include <iostream>
#include "P1.h"

#define N_EL 2
#define DIM (N_EL+1)*(N_EL+1)

using namespace dolfin;
using array_2d = std::vector<std::vector<double>>;


array_2d
build_mass_matrix(array_2d& F, std::vector<double>& K) {
    auto mesh = std::make_shared<UnitSquareMesh>(N_EL, N_EL);
    auto V = std::make_shared<P1::Form_M_FunctionSpace_0>(mesh);

    auto u = std::make_shared<Function>(V);
    auto v = std::make_shared<Function>(V);
    auto k = std::make_shared<Function>(V);
    k->vector()->set_local(K);

    P1::Form_M mass_form(mesh, k, u, v);
    array_2d mass(4*N_EL, std::vector<double>(4*N_EL));

    for (int i=0; i<4*N_EL; ++i) {
        u->vector()->set_local(F[i]);
        for (int j=i; j<4*N_EL; ++j) {
            v->vector()->set_local(F[j]);
            mass[i][j] = assemble(mass_form);
        }
    }
    return mass;
}


array_2d
build_stiffness_matrix(array_2d& F, std::vector<double>& K) {
    auto mesh = std::make_shared<UnitSquareMesh>(N_EL, N_EL);
    auto V = std::make_shared<P1::Form_S_FunctionSpace_0>(mesh);

    auto u = std::make_shared<Function>(V);
    auto v = std::make_shared<Function>(V);
    auto k = std::make_shared<Function>(V);
    k->vector()->set_local(K);

    P1::Form_S stiffness_form(mesh, k, u, v);
    array_2d stiffness(4*N_EL, std::vector<double>(4*N_EL));

    for (int i=0; i<4*N_EL; ++i) {
        u->vector()->set_local(F[i]);
        for (int j=i; j<4*N_EL; ++j) {
            v->vector()->set_local(F[j]);
            stiffness[i][j] = assemble(stiffness_form);
        }
    }
    return stiffness;
}

template <class T>
void print_vector(std::vector<T> vec) {
    for (auto const& v: vec) std::cout << v << '\t';
    std::cout << '\n';
}

template <class T>
void print_matrix(array_2d mat) {
    for (auto const& row: mat) print_vector<T>(row);
}

int main () {
    // Simple tests
    std::vector<double> K(DIM, 1);
    array_2d F(4*N_EL, K);
    std::cout << "vector K\n";
    print_vector<double>(K);
    std::vector<double> row;
    for (int i=0; i<4*N_EL; ++i) {
        row.assign(DIM, (double)i);
        F[i] = row;
    }
    std::cout << "matrix F\n";
    print_matrix<double>(F);

    std::cout << "mass matrix\n";
    auto mmat = build_mass_matrix(F, K);
    print_matrix<double>(mmat);
    std::cout << "stiffness matrix\n";
    auto smat = build_stiffness_matrix(F, K);
    print_matrix<double>(smat);
}
