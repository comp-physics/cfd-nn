/// @file turbulence_les_dynamic_gpu.cpp
#include "turbulence_device_view.hpp"
#ifdef USE_GPU_OFFLOAD
#include <omp.h>
#endif
namespace nncfd {
void dsmag_init_gpu_buffers(double*& ucc, double*& vcc, double*& wcc,
                             double*& lm, double*& mm, double*& cs2,
                             int cell_total, int Ny) {
    ucc = new double[cell_total]();
    vcc = new double[cell_total]();
    wcc = new double[cell_total]();
    lm = new double[Ny]();
    mm = new double[Ny]();
    cs2 = new double[Ny]();
    #pragma omp target enter data map(alloc: ucc[0:cell_total], \
        vcc[0:cell_total], wcc[0:cell_total], lm[0:Ny], mm[0:Ny], cs2[0:Ny])
}
void dsmag_cleanup_gpu_buffers(double*& ucc, double*& vcc, double*& wcc,
                                double*& lm, double*& mm, double*& cs2,
                                int cell_total, int Ny, bool gpu_ready) {
    if (gpu_ready) {
        [[maybe_unused]] int ct = cell_total;
        [[maybe_unused]] int ny = Ny;
        #pragma omp target exit data map(delete: ucc[0:ct], vcc[0:ct], wcc[0:ct], \
            lm[0:ny], mm[0:ny], cs2[0:ny])
    }
    delete[] ucc; ucc = nullptr;
    delete[] vcc; vcc = nullptr;
    delete[] wcc; wcc = nullptr;
    delete[] lm; lm = nullptr;
    delete[] mm; mm = nullptr;
    delete[] cs2; cs2 = nullptr;
}
} // namespace nncfd
