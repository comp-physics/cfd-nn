/// @file momentum_solver_hypre.cpp
/// @brief HYPRE-based momentum solver for GPU SIMPLE
///
/// Uses HYPRE's BiCGSTAB + PFMG preconditioner for the momentum equation.
/// PFMG (semicoarsening multigrid) is optimal for structured grids with
/// variable coefficients. BiCGSTAB handles the non-symmetry from upwind
/// convection.

#ifdef USE_HYPRE

#include "momentum_solver_hypre.hpp"
#include <stdexcept>
#include <cstring>

namespace nncfd {

HypreMomentumSolver::HypreMomentumSolver(const Mesh& mesh, bool is_u_component)
    : mesh_(&mesh), is_u_component_(is_u_component)
{
    // Grid dimensions for this velocity component
    // u-momentum: (Nx+1) × Ny cells on a staggered grid
    // v-momentum: Nx × (Ny+1) cells
    if (is_u_component_) {
        nx_ = mesh.Nx + 1;
        ny_ = mesh.Ny;
    } else {
        nx_ = mesh.Nx;
        ny_ = mesh.Ny + 1;
    }
    nz_ = mesh.is2D() ? 1 : mesh.Nz;
    n_cells_ = static_cast<size_t>(nx_) * ny_ * nz_;

    ilower_[0] = 0;
    ilower_[1] = 0;
    ilower_[2] = 0;
    iupper_[0] = nx_ - 1;
    iupper_[1] = ny_ - 1;
    iupper_[2] = nz_ > 1 ? nz_ - 1 : 0;

    rhs_host_.resize(n_cells_);
    x_host_.resize(n_cells_);
    coeff_host_.resize(n_cells_);

    create_grid();
    create_stencil();
    create_matrix();
    create_vectors();
    create_solver();

    initialized_ = true;
}

HypreMomentumSolver::~HypreMomentumSolver() {
    if (precond_) HYPRE_StructPFMGDestroy(precond_);
    if (solver_) HYPRE_StructBiCGSTABDestroy(solver_);
    if (x_) HYPRE_StructVectorDestroy(x_);
    if (b_) HYPRE_StructVectorDestroy(b_);
    if (A_) HYPRE_StructMatrixDestroy(A_);
    if (stencil_) HYPRE_StructStencilDestroy(stencil_);
    if (grid_) HYPRE_StructGridDestroy(grid_);
}

void HypreMomentumSolver::create_grid() {
    const int ndim = mesh_->is2D() ? 2 : 3;
    HYPRE_StructGridCreate(MPI_COMM_SELF, ndim, &grid_);
    HYPRE_StructGridSetExtents(grid_, ilower_, iupper_);

    // Periodic x for channel/duct
    HYPRE_Int period[3] = {0, 0, 0};
    period[0] = nx_;  // periodic in x (streamwise) for both u and v
    HYPRE_StructGridSetPeriodic(grid_, period);
    HYPRE_StructGridAssemble(grid_);
}

void HypreMomentumSolver::create_stencil() {
    const int ndim = mesh_->is2D() ? 2 : 3;
    const int stencil_size = ndim == 2 ? 5 : 7;
    HYPRE_StructStencilCreate(ndim, stencil_size, &stencil_);

    HYPRE_Int offsets[7][3] = {
        { 0,  0,  0},  // center (0)
        {-1,  0,  0},  // west   (1)
        { 1,  0,  0},  // east   (2)
        { 0, -1,  0},  // south  (3)
        { 0,  1,  0},  // north  (4)
        { 0,  0, -1},  // back   (5) — 3D only
        { 0,  0,  1},  // front  (6) — 3D only
    };
    for (int i = 0; i < stencil_size; ++i) {
        HYPRE_StructStencilSetElement(stencil_, i, offsets[i]);
    }
}

void HypreMomentumSolver::create_matrix() {
    HYPRE_StructMatrixCreate(MPI_COMM_SELF, grid_, stencil_, &A_);
    HYPRE_StructMatrixInitialize(A_);
}

void HypreMomentumSolver::create_vectors() {
    HYPRE_StructVectorCreate(MPI_COMM_SELF, grid_, &b_);
    HYPRE_StructVectorCreate(MPI_COMM_SELF, grid_, &x_);
    HYPRE_StructVectorInitialize(b_);
    HYPRE_StructVectorInitialize(x_);
}

void HypreMomentumSolver::create_solver() {
    // BiCGSTAB for the non-symmetric momentum system
    HYPRE_StructBiCGSTABCreate(MPI_COMM_SELF, &solver_);
    HYPRE_StructBiCGSTABSetMaxIter(solver_, 20);
    HYPRE_StructBiCGSTABSetTol(solver_, 1e-4);
    HYPRE_StructBiCGSTABSetLogging(solver_, 0);

    // PFMG preconditioner (structured multigrid)
    HYPRE_StructPFMGCreate(MPI_COMM_SELF, &precond_);
    HYPRE_StructPFMGSetMaxIter(precond_, 1);  // 1 V-cycle per preconditioning step
    HYPRE_StructPFMGSetTol(precond_, 0.0);     // exact preconditioning
    HYPRE_StructPFMGSetNumPreRelax(precond_, 1);
    HYPRE_StructPFMGSetNumPostRelax(precond_, 1);

    HYPRE_StructBiCGSTABSetPrecond(solver_,
        HYPRE_StructPFMGSolve, HYPRE_StructPFMGSetup, precond_);
}

void HypreMomentumSolver::set_coefficients(
    const double* a_W, const double* a_E,
    const double* a_S, const double* a_N,
    const double* a_B, const double* a_F,
    const double* a_P, int n_cells)
{
    // Set matrix coefficients for each stencil entry
    // Stencil: 0=center, 1=west, 2=east, 3=south, 4=north, 5=back, 6=front
    const int stencil_size = mesh_->is2D() ? 5 : 7;

    for (int s = 0; s < stencil_size; ++s) {
        HYPRE_Int stencil_indices[1] = {s};

        for (size_t i = 0; i < n_cells_; ++i) {
            switch (s) {
                case 0: coeff_host_[i] =  a_P[i]; break;
                case 1: coeff_host_[i] = -a_W[i]; break;
                case 2: coeff_host_[i] = -a_E[i]; break;
                case 3: coeff_host_[i] = -a_S[i]; break;
                case 4: coeff_host_[i] = -a_N[i]; break;
                case 5: coeff_host_[i] = a_B ? -a_B[i] : 0.0; break;
                case 6: coeff_host_[i] = a_F ? -a_F[i] : 0.0; break;
            }
        }

        HYPRE_StructMatrixSetBoxValues(A_, ilower_, iupper_,
            1, stencil_indices, coeff_host_.data());
    }

    HYPRE_StructMatrixAssemble(A_);

    // Re-setup the solver with the new matrix
    HYPRE_StructBiCGSTABSetup(solver_, A_, b_, x_);
}

int HypreMomentumSolver::solve(const double* b, double* x,
                                 double tol, int max_iter)
{
    HYPRE_StructBiCGSTABSetTol(solver_, tol);
    HYPRE_StructBiCGSTABSetMaxIter(solver_, max_iter);

    // Pack RHS (strip ghost cells from the solver's field layout)
    std::memcpy(rhs_host_.data(), b, n_cells_ * sizeof(double));
    HYPRE_StructVectorSetBoxValues(b_, ilower_, iupper_, rhs_host_.data());
    HYPRE_StructVectorAssemble(b_);

    // Pack initial guess
    std::memcpy(x_host_.data(), x, n_cells_ * sizeof(double));
    HYPRE_StructVectorSetBoxValues(x_, ilower_, iupper_, x_host_.data());
    HYPRE_StructVectorAssemble(x_);

    // Solve
    HYPRE_StructBiCGSTABSolve(solver_, A_, b_, x_);

    // Extract solution
    HYPRE_StructVectorGetBoxValues(x_, ilower_, iupper_, x_host_.data());
    std::memcpy(x, x_host_.data(), n_cells_ * sizeof(double));

    // Get convergence info
    HYPRE_Int num_iter;
    HYPRE_StructBiCGSTABGetNumIterations(solver_, &num_iter);
    HYPRE_StructBiCGSTABGetFinalRelativeResidualNorm(solver_, &final_residual_);

    return static_cast<int>(num_iter);
}

} // namespace nncfd

#endif // USE_HYPRE
