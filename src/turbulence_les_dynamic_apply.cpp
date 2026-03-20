/// @file turbulence_les_dynamic_apply.cpp
#include "turbulence_device_view.hpp"
#include <cmath>
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif
namespace nncfd {
#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif
inline void tg3c(int ig, int jg, int kg, double dx, double dz, const double* yc,
    const double* u, int us, int up, const double* v, int vs, int vp,
    const double* w, int ws, int wp, double g[9]) {
    double dyc=yc[jg+1]-yc[jg-1];
    g[0]=(u[kg*up+jg*us+ig+1]-u[kg*up+jg*us+ig])/dx;
    g[1]=(0.5*(u[kg*up+(jg+1)*us+ig]+u[kg*up+(jg+1)*us+ig+1])-0.5*(u[kg*up+(jg-1)*us+ig]+u[kg*up+(jg-1)*us+ig+1]))/dyc;
    g[2]=(0.5*(u[(kg+1)*up+jg*us+ig]+u[(kg+1)*up+jg*us+ig+1])-0.5*(u[(kg-1)*up+jg*us+ig]+u[(kg-1)*up+jg*us+ig+1]))/(2*dz);
    g[3]=(0.5*(v[kg*vp+jg*vs+ig+1]+v[kg*vp+(jg+1)*vs+ig+1])-0.5*(v[kg*vp+jg*vs+ig-1]+v[kg*vp+(jg+1)*vs+ig-1]))/(2*dx);
    g[4]=(v[kg*vp+(jg+1)*vs+ig]-v[kg*vp+jg*vs+ig])/(yc[jg+1]-yc[jg]);
    g[5]=(0.5*(v[(kg+1)*vp+jg*vs+ig]+v[(kg+1)*vp+(jg+1)*vs+ig])-0.5*(v[(kg-1)*vp+jg*vs+ig]+v[(kg-1)*vp+(jg+1)*vs+ig]))/(2*dz);
    g[6]=(0.5*(w[kg*wp+jg*ws+ig+1]+w[(kg+1)*wp+jg*ws+ig+1])-0.5*(w[kg*wp+jg*ws+ig-1]+w[(kg+1)*wp+jg*ws+ig-1]))/(2*dx);
    g[7]=(0.5*(w[kg*wp+(jg+1)*ws+ig]+w[(kg+1)*wp+(jg+1)*ws+ig])-0.5*(w[kg*wp+(jg-1)*ws+ig]+w[(kg+1)*wp+(jg-1)*ws+ig]))/dyc;
    g[8]=(w[(kg+1)*wp+jg*ws+ig]-w[kg*wp+jg*ws+ig])/dz;
}
#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif
void dsmag_pass2_apply(const TurbulenceDeviceView* dv,
    double* lm_plane, double* mm_plane, double* cs2_plane, int ny_sz) {
    const int Nx=dv->Nx, Ny=dv->Ny, Ng=dv->Ng, Nz=dv->Nz;
    const int total=Nx*Ny*Nz;
    const double dx=dv->dx, dz=dv->dz;
    double *u=dv->u_face, *v=dv->v_face, *w=dv->w_face, *nut=dv->nu_t;
    const double *yf=dv->yf, *yc=dv->yc;
    const int us=dv->u_stride,vs=dv->v_stride,ws=dv->w_stride;
    const int up=dv->u_plane_stride,vp=dv->v_plane_stride,wp=dv->w_plane_stride;
    const int cs=dv->cell_stride, cp=dv->cell_plane_stride;
    [[maybe_unused]] const int usz=dv->u_total,vsz=dv->v_total,wsz=dv->w_total;
    [[maybe_unused]] const int nsz=dv->cell_total;
    [[maybe_unused]] const int yfsz=dv->yf_total,ycsz=dv->yc_total;
    const double Cs2_max=0.5;

    #pragma omp target teams distribute parallel for \
        map(present: lm_plane[0:ny_sz],mm_plane[0:ny_sz],cs2_plane[0:ny_sz]) \
        firstprivate(Ny,Cs2_max)
    for (int j=0; j<Ny; ++j) {
        double cs2=(mm_plane[j]>1e-30)?lm_plane[j]/mm_plane[j]:0.0;
        if (cs2<0.0) cs2=0.0;
        if (cs2>Cs2_max) cs2=Cs2_max;
        cs2_plane[j]=cs2;
    }

    #pragma omp target teams distribute parallel for \
        map(present: u[0:usz],v[0:vsz],w[0:wsz],nut[0:nsz], \
            yf[0:yfsz],yc[0:ycsz],cs2_plane[0:ny_sz]) \
        firstprivate(Nx,Ny,Nz,Ng,dx,dz,us,vs,ws,up,vp,wp,cs,cp)
    for (int idx=0; idx<total; ++idx) {
        int kk=idx/(Nx*Ny), rem=idx-kk*Nx*Ny, j=rem/Nx, i=rem-j*Nx;
        int ig=i+Ng, jg=j+Ng, kg=kk+Ng;
        double g[9];
        tg3c(ig,jg,kg,dx,dz,yc,u,us,up,v,vs,vp,w,ws,wp,g);
        double S11=g[0],S22=g[4],S33=g[8];
        double S12=0.5*(g[1]+g[3]),S13=0.5*(g[2]+g[6]),S23=0.5*(g[5]+g[7]);
        double Sm=std::sqrt(2.0*(S11*S11+S22*S22+S33*S33+2.0*(S12*S12+S13*S13+S23*S23)));
        double dy=yf[jg+1]-yf[jg];
        double d=std::cbrt(dx*dy*dz);
        nut[kg*cp+jg*cs+ig]=cs2_plane[j]*d*d*Sm;
    }
}
} // namespace nncfd
