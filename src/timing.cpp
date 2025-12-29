#include "timing.hpp"
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <vector>

namespace nncfd {

// ============================================================================
// GPU utilization tracking implementation
// ============================================================================

bool TimingStats::is_gpu_category(const std::string& name) {
    // Categories ending in "_gpu" are explicitly GPU kernels
    const std::string gpu_suffix = "_gpu";
    if (name.length() >= gpu_suffix.length() &&
        name.compare(name.length() - gpu_suffix.length(), gpu_suffix.length(), gpu_suffix) == 0) {
        return true;
    }

    // For GPU builds, core solver compute operations run on GPU via OpenMP target
    // These are GPU categories unless explicitly marked as "_cpu"
#ifdef USE_GPU_OFFLOAD
    const std::string cpu_suffix = "_cpu";
    if (name.length() >= cpu_suffix.length() &&
        name.compare(name.length() - cpu_suffix.length(), cpu_suffix.length(), cpu_suffix) == 0) {
        return false;  // Explicitly CPU
    }

    // Core solver operations that run on GPU via unified kernels
    static const std::vector<std::string> gpu_compute_patterns = {
        "solver_step", "convective", "diffusive", "divergence", "poisson",
        "velocity_correction", "apply_bc", "bc_", "correct_",
        "turbulence_transport", "turbulence_update",
        "sst_transport", "komega_transport",
        "gradients", "closure"
    };

    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    for (const auto& pattern : gpu_compute_patterns) {
        if (lower_name.find(pattern) != std::string::npos) {
            return true;
        }
    }
#endif

    return false;
}

bool TimingStats::is_cpu_compute_category(const std::string& name) {
    // If it's already classified as GPU, it's not CPU compute
    if (is_gpu_category(name)) {
        return false;
    }

    // Exclude I/O and initialization categories
    static const std::vector<std::string> io_patterns = {
        "io_", "write_", "read_", "load_", "save_", "output_", "init_",
        "setup_", "allocate_", "deallocate_", "sync_", "upload", "download"
    };

    std::string lower_name = name;
    std::transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

    for (const auto& pattern : io_patterns) {
        if (lower_name.find(pattern) != std::string::npos) {
            return false;
        }
    }

    // Check for explicit CPU suffix - always CPU compute
    const std::string cpu_suffix = "_cpu";
    if (name.length() >= cpu_suffix.length() &&
        name.compare(name.length() - cpu_suffix.length(), cpu_suffix.length(), cpu_suffix) == 0) {
        return true;
    }

#ifdef USE_GPU_OFFLOAD
    // For GPU builds, most compute is on GPU; only explicitly CPU-marked
    // categories and non-compute categories are CPU
    // Check if this is a compute-related category that isn't GPU
    static const std::vector<std::string> compute_patterns = {
        "features", "inference", "postprocess", "mlp", "tbnn", "nn_", "gep", "earsm"
    };

    for (const auto& pattern : compute_patterns) {
        if (lower_name.find(pattern) != std::string::npos) {
            // This is compute but wasn't classified as GPU, so it's CPU compute
            return true;
        }
    }

    // For GPU builds, other categories are not CPU compute
    return false;
#else
    // For CPU builds, all compute categories are CPU compute
    static const std::vector<std::string> compute_patterns = {
        "solver_step", "convective", "diffusive", "divergence", "poisson",
        "velocity_correction", "turbulence", "transport", "gradients",
        "features", "inference", "postprocess", "closure", "mlp", "tbnn",
        "nn_", "gep", "earsm", "sst", "komega", "bc_"
    };

    for (const auto& pattern : compute_patterns) {
        if (lower_name.find(pattern) != std::string::npos) {
            return true;
        }
    }

    return false;
#endif
}

double TimingStats::gpu_kernel_time() const {
    double total = 0.0;
    for (const auto& [name, s] : stats_) {
        if (is_gpu_category(name)) {
            total += s.total_time;
        }
    }
    return total;
}

double TimingStats::cpu_compute_time() const {
    double total = 0.0;
    for (const auto& [name, s] : stats_) {
        if (is_cpu_compute_category(name)) {
            total += s.total_time;
        }
    }
    return total;
}

double TimingStats::total_compute_time() const {
    return gpu_kernel_time() + cpu_compute_time();
}

double TimingStats::gpu_utilization_ratio() const {
    double total = total_compute_time();
    if (total <= 0.0) {
        return 0.0;
    }
    return gpu_kernel_time() / total;
}

bool TimingStats::is_gpu_dominant(double threshold) const {
    return gpu_utilization_ratio() >= threshold;
}

void TimingStats::print_gpu_utilization_summary(std::ostream& os) const {
    double gpu_time = gpu_kernel_time();
    double cpu_time = cpu_compute_time();
    double total = total_compute_time();
    double ratio = gpu_utilization_ratio();

    os << "\n=== GPU Utilization Summary ===\n";
    os << std::fixed << std::setprecision(3);
    os << "GPU kernel time:     " << std::setw(10) << gpu_time << " s\n";
    os << "CPU compute time:    " << std::setw(10) << cpu_time << " s\n";
    os << "Total compute time:  " << std::setw(10) << total << " s\n";
    os << "GPU utilization:     " << std::setw(10) << (ratio * 100.0) << " %\n";
    os << std::string(40, '-') << "\n";

    // List GPU categories
    os << "\nGPU kernel breakdown:\n";
    for (const auto& [name, s] : stats_) {
        if (is_gpu_category(name)) {
            os << "  " << std::setw(30) << std::left << name
               << std::setw(10) << std::right << s.total_time << " s"
               << " (" << s.num_calls << " calls)\n";
        }
    }

    // List CPU compute categories
    os << "\nCPU compute breakdown:\n";
    for (const auto& [name, s] : stats_) {
        if (is_cpu_compute_category(name)) {
            os << "  " << std::setw(30) << std::left << name
               << std::setw(10) << std::right << s.total_time << " s"
               << " (" << s.num_calls << " calls)\n";
        }
    }
    os << "\n";
}

void TimingStats::assert_gpu_dominant(double threshold, const std::string& context) const {
    double ratio = gpu_utilization_ratio();
    if (ratio < threshold) {
        std::ostringstream oss;
        oss << "GPU utilization check FAILED";
        if (!context.empty()) {
            oss << " (" << context << ")";
        }
        oss << ": " << std::fixed << std::setprecision(1)
            << (ratio * 100.0) << "% < " << (threshold * 100.0) << "% threshold\n";
        oss << "GPU kernel time: " << gpu_kernel_time() << " s\n";
        oss << "CPU compute time: " << cpu_compute_time() << " s\n";
        throw std::runtime_error(oss.str());
    }
}

ScopedTimer::ScopedTimer(const std::string& name, bool print_on_exit)
    : name_(name), start_(std::chrono::steady_clock::now()), print_on_exit_(print_on_exit) {}

ScopedTimer::~ScopedTimer() {
    if (!stopped_ && print_on_exit_) {
        double t = elapsed();
        std::cout << "[Timer] " << name_ << ": " << t << " s\n";
    }
}

double ScopedTimer::elapsed() const {
    if (stopped_) {
        return elapsed_time_;
    }
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - start_;
    return diff.count();
}

double ScopedTimer::stop() {
    if (!stopped_) {
        elapsed_time_ = elapsed();
        stopped_ = true;
    }
    return elapsed_time_;
}

TimingStats& TimingStats::instance() {
    static TimingStats instance;
    return instance;
}

void TimingStats::record(const std::string& name, double seconds) {
    auto& s = stats_[name];
    s.total_time += seconds;
    s.num_calls++;
}

double TimingStats::total(const std::string& name) const {
    auto it = stats_.find(name);
    return it != stats_.end() ? it->second.total_time : 0.0;
}

int TimingStats::count(const std::string& name) const {
    auto it = stats_.find(name);
    return it != stats_.end() ? it->second.num_calls : 0;
}

double TimingStats::average(const std::string& name) const {
    auto it = stats_.find(name);
    if (it != stats_.end() && it->second.num_calls > 0) {
        return it->second.total_time / it->second.num_calls;
    }
    return 0.0;
}

void TimingStats::reset() {
    stats_.clear();
}

void TimingStats::print_summary(std::ostream& os) const {
    os << "\n=== Timing Summary ===\n";
    os << std::setw(30) << std::left << "Category"
       << std::setw(15) << std::right << "Total (s)"
       << std::setw(10) << "Calls"
       << std::setw(15) << "Avg (ms)"
       << "\n";
    os << std::string(70, '-') << "\n";
    
    for (const auto& [name, s] : stats_) {
        os << std::setw(30) << std::left << name
           << std::setw(15) << std::right << std::fixed << std::setprecision(3) << s.total_time
           << std::setw(10) << s.num_calls
           << std::setw(15) << std::setprecision(3) << (s.num_calls > 0 ? 1000.0 * s.total_time / s.num_calls : 0.0)
           << "\n";
    }
    os << std::string(70, '-') << "\n";
}

AutoTimer::AutoTimer(const std::string& name)
    : name_(name), start_(std::chrono::steady_clock::now()) {}

AutoTimer::~AutoTimer() {
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - start_;
    TimingStats::instance().record(name_, diff.count());
}

} // namespace nncfd


