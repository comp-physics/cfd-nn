/// @file turbulence_les_dynamic_interp.cpp
#include "turbulence_device_view.hpp"
#include <cmath>
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif
namespace nncfd {
void dsmag_pass0_interpolate(const TurbulenceDeviceView* dv,
    double* ucc, double* vcc, double* wcc, int cc_sz) {
    const int Nx = dv->Nx, Ny = dv->Ny, Ng = dv->Ng;
    const int Nz = dv->is3D() ? dv->Nz : 1;
    const int total = Nx * Ny * Nz;
    double* u = dv->u_face;
    double* v = dv->v_face;
    double* w = dv->w_face;
    const int us = dv->u_stride, vs = dv->v_stride, ws = dv->w_stride;
    const int up = dv->u_plane_stride, vp = dv->v_plane_stride, wp = dv->w_plane_stride;
    const int cs = dv->cell_stride, cp = dv->cell_plane_stride;
    [[maybe_unused]] const int usz = dv->u_total;
    [[maybe_unused]] const int vsz = dv->v_total;
    [[maybe_unused]] const int wsz = dv->w_total;
    const bool is2D = (Nz == 1);

    #pragma omp target teams distribute parallel for \
        map(present: u[0:usz], v[0:vsz], w[0:wsz], \
            ucc[0:cc_sz], vcc[0:cc_sz], wcc[0:cc_sz]) \
        firstprivate(Nx, Ny, Nz, Ng, is2D, us, vs, ws, up, vp, wp, cs, cp)
    for (int idx = 0; idx < total; ++idx) {
        int kk = idx / (Nx * Ny);
        int rem = idx - kk * Nx * Ny;
        int j = rem / Nx;
        int i = rem - j * Nx;
        int ig = i + Ng, jg = j + Ng, kg = kk + Ng;
        int ci = is2D ? (jg * cs + ig) : (kg * cp + jg * cs + ig);
        if (is2D) {
            ucc[ci] = 0.5 * (u[jg*us+ig] + u[jg*us+ig+1]);
            vcc[ci] = 0.5 * (v[jg*vs+ig] + v[(jg+1)*vs+ig]);
            wcc[ci] = 0.0;
        } else {
            ucc[ci] = 0.5 * (u[kg*up+jg*us+ig] + u[kg*up+jg*us+ig+1]);
            vcc[ci] = 0.5 * (v[kg*vp+jg*vs+ig] + v[kg*vp+(jg+1)*vs+ig]);
            wcc[ci] = 0.5 * (w[kg*wp+jg*ws+ig] + w[(kg+1)*wp+jg*ws+ig]);
        }
    }
}
} // namespace nncfd
