/// @file velocity_gradient.cpp
/// @brief Velocity gradient tensor computation on staggered MAC grid

#include "velocity_gradient.hpp"
#include <algorithm>

namespace nncfd {

void GradientTensor3D::resize(int nx, int ny, int nz) {
    Nx = nx;
    Ny = ny;
    Nz = nz;
    int n = nx * ny * nz;
    g11.assign(n, 0.0); g12.assign(n, 0.0); g13.assign(n, 0.0);
    g21.assign(n, 0.0); g22.assign(n, 0.0); g23.assign(n, 0.0);
    g31.assign(n, 0.0); g32.assign(n, 0.0); g33.assign(n, 0.0);
}

void GradientComputer::compute(const Mesh& mesh, const VectorField& vel,
                                GradientTensor3D& grad) const {
    const int Nx = mesh.Nx;
    const int Ny = mesh.Ny;
    const int Nz = std::max(mesh.Nz, 1);
    const int Ng = mesh.Nghost;
    const bool is2D = mesh.is2D();
    int Nz_eff = is2D ? 1 : Nz;

    grad.resize(Nx, Ny, Nz_eff);

    for (int k = 0; k < Nz_eff; ++k) {
        int kg = k + Ng;
        for (int j = 0; j < Ny; ++j) {
            int jg = j + Ng;
            for (int i = 0; i < Nx; ++i) {
                int ig = i + Ng;
                int idx = grad.index(i, j, k);

                // du/dx: u is at x-faces, so du/dx at cell center = (u[i+1] - u[i]) / dx
                // On staggered grid: u(ig, jg) is at x_{ig-1/2}, u(ig+1, jg) is at x_{ig+1/2}
                // Cell center is at x_ig, so du/dx = (u(ig+1) - u(ig)) / dx
                if (is2D) {
                    double dudx = (vel.u(ig + 1, jg) - vel.u(ig, jg)) / mesh.dx;
                    grad.g11[idx] = dudx;

                    // du/dy: interpolate u to cell center first, then differentiate in y
                    // u at (ig, jg) center = 0.5*(u(ig, jg) + u(ig+1, jg))
                    double u_jp = 0.5 * (vel.u(ig, jg + 1) + vel.u(ig + 1, jg + 1));
                    double u_jm = 0.5 * (vel.u(ig, jg - 1) + vel.u(ig + 1, jg - 1));
                    grad.g12[idx] = (u_jp - u_jm) / (2.0 * mesh.dy);

                    // du/dz = 0 in 2D
                    grad.g13[idx] = 0.0;

                    // dv/dx: interpolate v to cell center, differentiate in x
                    double v_ip = 0.5 * (vel.v(ig + 1, jg) + vel.v(ig + 1, jg + 1));
                    double v_im = 0.5 * (vel.v(ig - 1, jg) + vel.v(ig - 1, jg + 1));
                    grad.g21[idx] = (v_ip - v_im) / (2.0 * mesh.dx);

                    // dv/dy: v is at y-faces, so dv/dy = (v(jg+1) - v(jg)) / dy
                    double dvdy = (vel.v(ig, jg + 1) - vel.v(ig, jg)) / mesh.dy;
                    grad.g22[idx] = dvdy;

                    // dv/dz = 0 in 2D
                    grad.g23[idx] = 0.0;

                    // All w-gradients zero in 2D
                    grad.g31[idx] = 0.0;
                    grad.g32[idx] = 0.0;
                    grad.g33[idx] = 0.0;
                } else {
                    // 3D case
                    // du/dx
                    grad.g11[idx] = (vel.u(ig + 1, jg, kg) - vel.u(ig, jg, kg)) / mesh.dx;

                    // du/dy (interpolate u to center, diff in y)
                    double u_jp = 0.5 * (vel.u(ig, jg + 1, kg) + vel.u(ig + 1, jg + 1, kg));
                    double u_jm = 0.5 * (vel.u(ig, jg - 1, kg) + vel.u(ig + 1, jg - 1, kg));
                    grad.g12[idx] = (u_jp - u_jm) / (2.0 * mesh.dy);

                    // du/dz (interpolate u to center, diff in z)
                    double u_kp = 0.5 * (vel.u(ig, jg, kg + 1) + vel.u(ig + 1, jg, kg + 1));
                    double u_km = 0.5 * (vel.u(ig, jg, kg - 1) + vel.u(ig + 1, jg, kg - 1));
                    grad.g13[idx] = (u_kp - u_km) / (2.0 * mesh.dz);

                    // dv/dx (interpolate v to center, diff in x)
                    double v_ip = 0.5 * (vel.v(ig + 1, jg, kg) + vel.v(ig + 1, jg + 1, kg));
                    double v_im = 0.5 * (vel.v(ig - 1, jg, kg) + vel.v(ig - 1, jg + 1, kg));
                    grad.g21[idx] = (v_ip - v_im) / (2.0 * mesh.dx);

                    // dv/dy
                    grad.g22[idx] = (vel.v(ig, jg + 1, kg) - vel.v(ig, jg, kg)) / mesh.dy;

                    // dv/dz (interpolate v to center, diff in z)
                    double v_kp = 0.5 * (vel.v(ig, jg, kg + 1) + vel.v(ig, jg + 1, kg + 1));
                    double v_km = 0.5 * (vel.v(ig, jg, kg - 1) + vel.v(ig, jg + 1, kg - 1));
                    grad.g23[idx] = (v_kp - v_km) / (2.0 * mesh.dz);

                    // dw/dx (interpolate w to center, diff in x)
                    double w_ip = 0.5 * (vel.w(ig + 1, jg, kg) + vel.w(ig + 1, jg, kg + 1));
                    double w_im = 0.5 * (vel.w(ig - 1, jg, kg) + vel.w(ig - 1, jg, kg + 1));
                    grad.g31[idx] = (w_ip - w_im) / (2.0 * mesh.dx);

                    // dw/dy (interpolate w to center, diff in y)
                    double w_jp = 0.5 * (vel.w(ig, jg + 1, kg) + vel.w(ig, jg + 1, kg + 1));
                    double w_jm = 0.5 * (vel.w(ig, jg - 1, kg) + vel.w(ig, jg - 1, kg + 1));
                    grad.g32[idx] = (w_jp - w_jm) / (2.0 * mesh.dy);

                    // dw/dz
                    grad.g33[idx] = (vel.w(ig, jg, kg + 1) - vel.w(ig, jg, kg)) / mesh.dz;
                }
            }
        }
    }
}

} // namespace nncfd
