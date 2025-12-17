#include "gpu_utils.hpp"
#include <stdexcept>
#include <omp.h>

namespace nncfd {
namespace gpu {

void verify_device_available() {
    int num_devices = omp_get_num_devices();
    if (num_devices == 0) {
        throw std::runtime_error(
            "GPU build requires GPU device. Found 0. "
            "Run on GPU node or rebuild with USE_GPU_OFFLOAD=OFF."
        );
    }
}

bool is_pointer_present(void* ptr) {
    return omp_target_is_present(ptr, omp_get_default_device());
}

} // namespace gpu
} // namespace nncfd
