/// @file turbulence_les_dynamic_germano.cpp
#include "turbulence_device_view.hpp"
#include <cmath>
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif
namespace nncfd {
#ifdef USE_GPU_OFFLOAD
#pragma omp declare target
#endif
inline double tbf(const double* f, int ig, int jg, int kg,
                  int s, int p, int Ny, int Ng) {
    double mlo = (jg-1 >= Ng) ? 1.0 : 0.0;
    double mhi = (jg+1 < Ng+Ny) ? 1.0 : 0.0;
    double r0 = mlo*(f[(kg-1)*p+(jg-1)*s+ig-1]+f[(kg-1)*p+(jg-1)*s+ig]+f[(kg-1)*p+(jg-1)*s+ig+1]
      +f[kg*p+(jg-1)*s+ig-1]+f[kg*p+(jg-1)*s+ig]+f[kg*p+(jg-1)*s+ig+1]
      +f[(kg+1)*p+(jg-1)*s+ig-1]+f[(kg+1)*p+(jg-1)*s+ig]+f[(kg+1)*p+(jg-1)*s+ig+1]);
    double r1 = f[(kg-1)*p+jg*s+ig-1]+f[(kg-1)*p+jg*s+ig]+f[(kg-1)*p+jg*s+ig+1]
      +f[kg*p+jg*s+ig-1]+f[kg*p+jg*s+ig]+f[kg*p+jg*s+ig+1]
      +f[(kg+1)*p+jg*s+ig-1]+f[(kg+1)*p+jg*s+ig]+f[(kg+1)*p+jg*s+ig+1];
    double r2 = mhi*(f[(kg-1)*p+(jg+1)*s+ig-1]+f[(kg-1)*p+(jg+1)*s+ig]+f[(kg-1)*p+(jg+1)*s+ig+1]
      +f[kg*p+(jg+1)*s+ig-1]+f[kg*p+(jg+1)*s+ig]+f[kg*p+(jg+1)*s+ig+1]
      +f[(kg+1)*p+(jg+1)*s+ig-1]+f[(kg+1)*p+(jg+1)*s+ig]+f[(kg+1)*p+(jg+1)*s+ig+1]);
    return (r0+r1+r2) / (9.0+9.0*mlo+9.0*mhi);
}
inline double tbfp(const double* a, const double* b, int ig, int jg, int kg,
                   int s, int p, int Ny, int Ng) {
    double mlo = (jg-1 >= Ng) ? 1.0 : 0.0;
    double mhi = (jg+1 < Ng+Ny) ? 1.0 : 0.0;
#define AB(K,J,I) (a[(K)*p+(J)*s+(I)]*b[(K)*p+(J)*s+(I)])
    double r0 = mlo*(AB(kg-1,jg-1,ig-1)+AB(kg-1,jg-1,ig)+AB(kg-1,jg-1,ig+1)
      +AB(kg,jg-1,ig-1)+AB(kg,jg-1,ig)+AB(kg,jg-1,ig+1)
      +AB(kg+1,jg-1,ig-1)+AB(kg+1,jg-1,ig)+AB(kg+1,jg-1,ig+1));
    double r1 = AB(kg-1,jg,ig-1)+AB(kg-1,jg,ig)+AB(kg-1,jg,ig+1)
      +AB(kg,jg,ig-1)+AB(kg,jg,ig)+AB(kg,jg,ig+1)
      +AB(kg+1,jg,ig-1)+AB(kg+1,jg,ig)+AB(kg+1,jg,ig+1);
    double r2 = mhi*(AB(kg-1,jg+1,ig-1)+AB(kg-1,jg+1,ig)+AB(kg-1,jg+1,ig+1)
      +AB(kg,jg+1,ig-1)+AB(kg,jg+1,ig)+AB(kg,jg+1,ig+1)
      +AB(kg+1,jg+1,ig-1)+AB(kg+1,jg+1,ig)+AB(kg+1,jg+1,ig+1));
#undef AB
    return (r0+r1+r2) / (9.0+9.0*mlo+9.0*mhi);
}
inline void tg3(int ig, int jg, int kg, double dx, double dz, const double* yc,
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
/// 2D box filter: 3x3 in x-y with y-wall truncation (no z-stencil)
inline double tbf2d(const double* f, int ig, int jg,
                    int s, int Ny, int Ng) {
    double mlo = (jg-1 >= Ng) ? 1.0 : 0.0;
    double mhi = (jg+1 < Ng+Ny) ? 1.0 : 0.0;
    double r0 = mlo * (f[(jg-1)*s+ig-1] + f[(jg-1)*s+ig] + f[(jg-1)*s+ig+1]);
    double r1 = f[jg*s+ig-1] + f[jg*s+ig] + f[jg*s+ig+1];
    double r2 = mhi * (f[(jg+1)*s+ig-1] + f[(jg+1)*s+ig] + f[(jg+1)*s+ig+1]);
    return (r0+r1+r2) / (3.0+3.0*mlo+3.0*mhi);
}
/// 2D box filter product
inline double tbfp2d(const double* a, const double* b, int ig, int jg,
                     int s, int Ny, int Ng) {
    double mlo = (jg-1 >= Ng) ? 1.0 : 0.0;
    double mhi = (jg+1 < Ng+Ny) ? 1.0 : 0.0;
#define AB2(J,I) (a[(J)*s+(I)]*b[(J)*s+(I)])
    double r0 = mlo * (AB2(jg-1,ig-1) + AB2(jg-1,ig) + AB2(jg-1,ig+1));
    double r1 = AB2(jg,ig-1) + AB2(jg,ig) + AB2(jg,ig+1);
    double r2 = mhi * (AB2(jg+1,ig-1) + AB2(jg+1,ig) + AB2(jg+1,ig+1));
#undef AB2
    return (r0+r1+r2) / (3.0+3.0*mlo+3.0*mhi);
}
/// 2D gradient computation
inline void tg2d(int ig, int jg, double dx, const double* yc,
    const double* u, int us, const double* v, int vs, double g[9]) {
    double dyc = yc[jg+1]-yc[jg-1];
    g[0] = (u[jg*us+ig+1]-u[jg*us+ig])/dx;
    g[1] = (0.5*(u[(jg+1)*us+ig]+u[(jg+1)*us+ig+1])
           -0.5*(u[(jg-1)*us+ig]+u[(jg-1)*us+ig+1]))/dyc;
    g[2] = 0;
    g[3] = (0.5*(v[jg*vs+ig+1]+v[(jg+1)*vs+ig+1])
           -0.5*(v[jg*vs+ig-1]+v[(jg+1)*vs+ig-1]))/(2*dx);
    g[4] = (v[(jg+1)*vs+ig]-v[jg*vs+ig])/(yc[jg+1]-yc[jg]);
    g[5]=0; g[6]=0; g[7]=0; g[8]=0;
}
inline double tfw(const double* yf, int jg, double dx, double dz) {
    return std::cbrt(dx*(yf[jg+1]-yf[jg])*dz);
}
inline double tfw2d(const double* yf, int jg, double dx) {
    return std::sqrt(dx*(yf[jg+1]-yf[jg]));
}
#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

void dsmag_pass1_germano(const TurbulenceDeviceView* dv,
    double* ucc, double* vcc, double* wcc,
    double* lm_plane, double* mm_plane,
    [[maybe_unused]] int cc_sz, [[maybe_unused]] int ny_sz) {
    const int Nx=dv->Nx, Ny=dv->Ny, Ng=dv->Ng;
    const int Nz_eff = (dv->Nz > 1) ? dv->Nz : 1;
    const bool is2D = (dv->Nz <= 1);
    const int total=Nx*Ny*Nz_eff;
    const double dx=dv->dx, dz=dv->dz;
    double *u=dv->u_face, *v=dv->v_face, *w=dv->w_face;
    const double *yf=dv->yf, *yc=dv->yc;
    const int us=dv->u_stride,vs=dv->v_stride,ws=dv->w_stride;
    const int up=dv->u_plane_stride,vp=dv->v_plane_stride,wp=dv->w_plane_stride;
    const int cs=dv->cell_stride, cp=dv->cell_plane_stride;
    [[maybe_unused]] const int usz=dv->u_total,vsz=dv->v_total,wsz=dv->w_total;
    [[maybe_unused]] const int yfsz=dv->yf_total,ycsz=dv->yc_total;

    #pragma omp target teams distribute parallel for \
        map(present: lm_plane[0:ny_sz], mm_plane[0:ny_sz]) firstprivate(Ny)
    for (int j=0; j<Ny; ++j) { lm_plane[j]=0; mm_plane[j]=0; }

    #pragma omp target teams distribute parallel for \
        map(present: u[0:usz],v[0:vsz],w[0:wsz], \
            ucc[0:cc_sz],vcc[0:cc_sz],wcc[0:cc_sz], \
            yf[0:yfsz],yc[0:ycsz],lm_plane[0:ny_sz],mm_plane[0:ny_sz]) \
        firstprivate(Nx,Ny,Nz_eff,Ng,dx,dz,is2D,us,vs,ws,up,vp,wp,cs,cp)
    for (int idx=0; idx<total; ++idx) {
        int kk=idx/(Nx*Ny), rem=idx-kk*Nx*Ny, j=rem/Nx, i=rem-j*Nx;
        int ig=i+Ng, jg=j+Ng, kg=kk+Ng;

        // Velocity gradients (2D or 3D)
        double g[9];
        if (is2D) {
            tg2d(ig, jg, dx, yc, u, us, v, vs, g);
        } else {
            tg3(ig,jg,kg,dx,dz,yc,u,us,up,v,vs,vp,w,ws,wp,g);
        }

        double S11=g[0],S22=g[4],S33=g[8];
        double S12=0.5*(g[1]+g[3]),S13=0.5*(g[2]+g[6]),S23=0.5*(g[5]+g[7]);
        double Sm=std::sqrt(2.0*(S11*S11+S22*S22+S33*S33+2.0*(S12*S12+S13*S13+S23*S23)));
        double d = is2D ? tfw2d(yf,jg,dx) : tfw(yf,jg,dx,dz);
        double fac=3.0*d*d*Sm;
        double M11=fac*S11,M22=fac*S22,M33=fac*S33,M12=fac*S12,M13=fac*S13,M23=fac*S23;

        // Box filter: 2D (3x3 in x-y) or 3D (3x3x3 in x-y-z)
        double ub,vb,wb,uu,uv,vv,uw,vw,ww;
        if (is2D) {
            ub = tbf2d(ucc,ig,jg,cs,Ny,Ng);
            vb = tbf2d(vcc,ig,jg,cs,Ny,Ng);
            wb = 0.0;
            uu = tbfp2d(ucc,ucc,ig,jg,cs,Ny,Ng);
            uv = tbfp2d(ucc,vcc,ig,jg,cs,Ny,Ng);
            vv = tbfp2d(vcc,vcc,ig,jg,cs,Ny,Ng);
            uw=0; vw=0; ww=0;
        } else {
            ub=tbf(ucc,ig,jg,kg,cs,cp,Ny,Ng);
            vb=tbf(vcc,ig,jg,kg,cs,cp,Ny,Ng);
            wb=tbf(wcc,ig,jg,kg,cs,cp,Ny,Ng);
            uu=tbfp(ucc,ucc,ig,jg,kg,cs,cp,Ny,Ng);
            uv=tbfp(ucc,vcc,ig,jg,kg,cs,cp,Ny,Ng);
            vv=tbfp(vcc,vcc,ig,jg,kg,cs,cp,Ny,Ng);
            uw=tbfp(ucc,wcc,ig,jg,kg,cs,cp,Ny,Ng);
            vw=tbfp(vcc,wcc,ig,jg,kg,cs,cp,Ny,Ng);
            ww=tbfp(wcc,wcc,ig,jg,kg,cs,cp,Ny,Ng);
        }

        double L11=uu-ub*ub,L22=vv-vb*vb,L33=ww-wb*wb;
        double L12=uv-ub*vb,L13=uw-ub*wb,L23=vw-vb*wb;
        double LM=L11*M11+L22*M22+L33*M33+2.0*(L12*M12+L13*M13+L23*M23);
        double MM=M11*M11+M22*M22+M33*M33+2.0*(M12*M12+M13*M13+M23*M23);
        #pragma omp atomic update
        lm_plane[j] += LM;
        #pragma omp atomic update
        mm_plane[j] += MM;
    }
}
} // namespace nncfd
