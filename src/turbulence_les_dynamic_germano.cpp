/// @file turbulence_les_dynamic_germano.cpp
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
inline double tfw(const double* yf, int jg, double dx, double dz) {
    return std::cbrt(dx*(yf[jg+1]-yf[jg])*dz);
}
#ifdef USE_GPU_OFFLOAD
#pragma omp end declare target
#endif

void dsmag_pass1_germano(const TurbulenceDeviceView* dv,
    double* ucc, double* vcc, double* wcc,
    double* lm_plane, double* mm_plane, int cc_sz, int ny_sz) {
    const int Nx=dv->Nx, Ny=dv->Ny, Ng=dv->Ng, Nz=dv->Nz;
    const int total=Nx*Ny*Nz;
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
        firstprivate(Nx,Ny,Nz,Ng,dx,dz,us,vs,ws,up,vp,wp,cs,cp)
    for (int idx=0; idx<total; ++idx) {
        int kk=idx/(Nx*Ny), rem=idx-kk*Nx*Ny, j=rem/Nx, i=rem-j*Nx;
        int ig=i+Ng, jg=j+Ng, kg=kk+Ng;
        double g[9];
        tg3(ig,jg,kg,dx,dz,yc,u,us,up,v,vs,vp,w,ws,wp,g);
        double S11=g[0],S22=g[4],S33=g[8];
        double S12=0.5*(g[1]+g[3]),S13=0.5*(g[2]+g[6]),S23=0.5*(g[5]+g[7]);
        double Sm=std::sqrt(2.0*(S11*S11+S22*S22+S33*S33+2.0*(S12*S12+S13*S13+S23*S23)));
        double d=tfw(yf,jg,dx,dz);
        double fac=3.0*d*d*Sm;
        double M11=fac*S11,M22=fac*S22,M33=fac*S33,M12=fac*S12,M13=fac*S13,M23=fac*S23;
        double ub=tbf(ucc,ig,jg,kg,cs,cp,Ny,Ng);
        double vb=tbf(vcc,ig,jg,kg,cs,cp,Ny,Ng);
        double wb=tbf(wcc,ig,jg,kg,cs,cp,Ny,Ng);
        double uu=tbfp(ucc,ucc,ig,jg,kg,cs,cp,Ny,Ng);
        double uv=tbfp(ucc,vcc,ig,jg,kg,cs,cp,Ny,Ng);
        double vv=tbfp(vcc,vcc,ig,jg,kg,cs,cp,Ny,Ng);
        double uw=tbfp(ucc,wcc,ig,jg,kg,cs,cp,Ny,Ng);
        double vw=tbfp(vcc,wcc,ig,jg,kg,cs,cp,Ny,Ng);
        double ww=tbfp(wcc,wcc,ig,jg,kg,cs,cp,Ny,Ng);
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
