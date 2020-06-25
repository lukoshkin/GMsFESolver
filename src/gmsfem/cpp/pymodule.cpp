#include <iostream>
#include <dolfin.h>
#include "EPDE.h"
#include "GEP.h"
#include <pybind11/pybind11.h>

using namespace dolfin;
namespace py = pybind11;


class Alpha: public SubDomain {
  bool inside(const Array<double>& x, bool on_boundary) const
  { return on_boundary; }
};


class Delta: public SubDomain {
    const double x;
    const double y;
    const float tol;
    public:
      Delta(double x, double y, float tol): x(x), y(y), tol(tol) {}

    private:
      bool inside(const Array<double>& z, bool on_boundary) const
        { return near(z[0], x, tol) and near(z[1], y, tol); }
};


PETScMatrix
snapshot_space(std::shared_ptr<Function>& k, float tol) {
  auto V = k->function_space();
  auto comm = V->mesh()->mpi_comm();
  auto f = std::make_shared<Constant>(0.);
  EPDE::BilinearForm a(V, V);
  EPDE::LinearForm L(V);
  a.k = k;
  L.f = f;
  
  PETScMatrix A(comm);
  PETScVector b(comm);
  PETScVector u(comm);
  assemble(A, a);
  assemble(b, L);

  auto u0 = std::make_shared<Constant>(0.);
  auto u1 = std::make_shared<Constant>(1.);
  auto bn0 = std::make_shared<Alpha>();
  DirichletBC bc0(V, u0, bn0);

  BoundaryMesh bmesh (*V->mesh(), "local");
  auto x = bmesh.coordinates();
  size_t n_ver{bmesh.num_vertices()}, DIM{V->dim()};
  std::vector<PetscScalar> buf(n_ver*DIM);
  PetscScalar * ptr;

  for (uint i=0; i<n_ver; ++i) {
    auto bn1 = std::make_shared<Delta>(x[2*i], x[2*i+1], tol);
    DirichletBC bc1(V, u1, bn1, "pointwise");

    auto _A = PETScMatrix(A);
    auto _b = PETScVector(b);
    bc0.apply(_A, _b);
    bc1.apply(_A, _b);

    solve(_A, u, _b);
    VecGetArray(u.vec(), &ptr);
    std::copy(ptr, ptr+DIM, buf.data()+i*DIM);
  }
  
  Mat Nv;
  MatCreateDense(comm, n_ver, DIM, n_ver, DIM, buf.data(), &Nv);
  return PETScMatrix(Nv);
}



PYBIND11_MODULE(SIGNATURE, m) {
    m.doc() = "Faster calculations with C++";
    m.def(
            "snapshot_space",
            &snapshot_space,
            "Builds Snapshot space",
            py::arg("kappa"), py::arg("tol"));
}
