#include <numeric>
#include <dolfin.h>
#include "EPDE.h"
#include "GEP.h"

using namespace dolfin;
MPI_Comm comm = PETSC_COMM_SELF;

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
  size_t n_ver{bmesh.num_vertices()}, DIM{V->dim()};
  std::vector<PetscScalar> buf(n_ver*DIM);
  auto x = bmesh.coordinates();
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
  
  Mat _Nv;
  MatCreateDense(comm, n_ver, DIM, n_ver, DIM, buf.data(), &_Nv);
  auto Nv = PETScMatrix(_Nv);
  MatDestroy(&_Nv);
  return Nv;
}


std::pair<PETScMatrix, std::vector<double>> 
model_reduction(std::shared_ptr<Function>& k, Mat& Nv, uint n_eig) {
  auto V = k->function_space();
  EPDE::BilinearForm a(V, V);
  GEP::BilinearForm q(V, V);
  a.set_coefficient("k", k);
  q.set_coefficient("k", k);
  
  PETScMatrix Mu(comm), Su(comm);
  assemble(Mu, q);
  assemble(Su, a);

  //std::shared_ptr<PETScMatrix>.copy() - copy of matrix
  Mat _M{Mu.mat()}, _S{Su.mat()};
  MatRARt(_M, Nv, MAT_INPLACE_MATRIX, PETSC_DEFAULT, &_M);
  MatRARt(_S, Nv, MAT_INPLACE_MATRIX, PETSC_DEFAULT, &_S);
  auto M = std::make_shared<PETScMatrix>(_M);
  auto S = std::make_shared<PETScMatrix>(_S);

  SLEPcEigenSolver esolver(comm, M, S);
  esolver.parameters["problem_type"] = "gen_hermitian";
  esolver.parameters["spectrum"] = "largest magnitude";
  esolver.parameters["solver"] = "power";
  esolver.solve(n_eig);
    
  // r - real, i - imaginary
  // l - lambda (eigenvalue), x - eigenvector
  double li;
  PETScVector xr(comm), xi;
  std::vector<uintptr_t> cols{V->dim()};
  std::iota(cols.begin(), cols.end(), 0);
  std::vector<double> lr(n_eig), buf(V->dim());
  PETScMatrix R(comm);
  for (uint i=0; i<n_eig; ++i) {
    esolver.get_eigenpair(lr[i], li, xr, xi, i);
    xr.get_local(buf);
    R.setrow(i, cols, buf); 
  }
  return std::make_pair(R, lr);
}

//std::pair<PETScMatrix, std::vector<double>> 
//model_reduction(
//        std::shared_ptr<Function>& k,
//        std::shared_ptr<PETScMatrix>& Nv,
//        uint n_eig) {
//  auto V = k->function_space();
//  auto cols = arange(0, V->dim());
//  auto comm = V->mesh()->mpi_comm();
//  EPDE::BilinearForm a(V, V);
//  GEP::BilinearForm q(V, V);
//  a.set_coefficient("k", k);
//  q.set_coefficient("k", k);
//  
//  auto M = std::make_shared<PETScMatrix>(comm);
//  auto S = std::make_shared<PETScMatrix>(comm);
//  assemble(*M, q);
//  assemble(*S, a);
//
//  Mat MM{M->mat()}, SS{S->mat()};
//  MatRARt(M->mat(), Nv->mat(), MAT_INPLACE_MATRIX, PETSC_DEFAULT, &M->mat());
//  MatRARt(S->mat(), Nv->mat(), MAT_INPLACE_MATRIX, PETSC_DEFAULT, &S->mat());
//
//  SLEPcEigenSolver esolver(comm, M, S);
//  esolver.parameters["problem_type"] = "gen_hermitian";
//  esolver.parameters["spectrum"] = "largest magnitude";
//  esolver.parameters["solver"] = "lapack";
//  esolver.solve(n_eig);
//    
//  // r - real, i - imaginary
//  // l - lambda (eigenvalue), x - eigenvector
//  double li;
//  PETScVector xr(comm), xi;
//  std::vector<double> lr(n_eig), buf(V->dim());
//  PETScMatrix R(comm);
//  for (uint i=0; i<n_eig; ++i) {
//    esolver.get_eigenpair(lr[i], li, xr, xi, i);
//    xr.get_local(buf);
//    R.setrow(i, cols, buf); 
//  }
//  return std::make_pair(R, lr);
//}

int main(int argc, char **argv) {
  PetscInitialize(&argc,&argv,0,0);
  int nel = 4;
  auto mesh = std::make_shared<UnitSquareMesh>(MPI_COMM_SELF, nel, nel);
  auto V = std::make_shared<GEP::FunctionSpace>(mesh);
  auto k = std::make_shared<Function>(V);
  auto Nv = snapshot_space(k, 1./nel);
  PetscFinalize();
  //model_reduction(k, Nv, 8);
}
