#pragma once

#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

#ifdef USE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace nncfd {

/// Simple scoped timer that records elapsed time
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name, bool print_on_exit = false);
    ~ScopedTimer();
    
    /// Get elapsed time in seconds
    double elapsed() const;
    
    /// Stop timer and return elapsed time
    double stop();
    
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
    bool print_on_exit_;
    double elapsed_time_ = 0.0;
};

/// Global timing statistics collector
class TimingStats {
public:
    static TimingStats& instance();

    /// Record a timing measurement
    void record(const std::string& name, double seconds);

    /// Get total time for a category
    double total(const std::string& name) const;

    /// Get number of calls for a category
    int count(const std::string& name) const;

    /// Get average time per call
    double average(const std::string& name) const;

    /// Reset all statistics
    void reset();

    /// Print summary to stdout
    void print_summary(std::ostream& os = std::cout) const;

    // ========================================================================
    // GPU utilization tracking for CI verification
    // ========================================================================

    /// Get total time spent in GPU kernels (categories ending in "_gpu")
    double gpu_kernel_time() const;

    /// Get total time spent in CPU compute (categories with "_cpu" suffix
    /// or compute-related categories without "_gpu" suffix)
    double cpu_compute_time() const;

    /// Get total compute time (GPU + CPU compute, excluding I/O and setup)
    double total_compute_time() const;

    /// Get GPU utilization ratio (0.0 to 1.0)
    /// Returns gpu_kernel_time / total_compute_time
    double gpu_utilization_ratio() const;

    /// Check if GPU dominates computation (for CI validation)
    /// @param threshold Minimum required GPU utilization (default 0.8 = 80%)
    /// @return true if GPU utilization >= threshold
    bool is_gpu_dominant(double threshold = 0.8) const;

    /// Print GPU utilization summary for CI
    void print_gpu_utilization_summary(std::ostream& os = std::cout) const;

    /// Assert GPU utilization meets threshold (throws if not met)
    /// @param threshold Minimum required GPU utilization
    /// @param context Description for error message
    void assert_gpu_dominant(double threshold, const std::string& context = "") const;

private:
    TimingStats() = default;

    /// Check if a timing category is a GPU kernel
    static bool is_gpu_category(const std::string& name);

    /// Check if a timing category is CPU compute (not I/O)
    static bool is_cpu_compute_category(const std::string& name);

    struct Stats {
        double total_time = 0.0;
        int num_calls = 0;
    };
    std::map<std::string, Stats> stats_;
};

/// RAII timer that automatically records to TimingStats
class AutoTimer {
public:
    explicit AutoTimer(const std::string& name);
    ~AutoTimer();
    
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

/// RAII NVTX range (no timing, just nsys annotation)
class NvtxRange {
public:
    explicit NvtxRange([[maybe_unused]] const char* name) {
#ifdef USE_NVTX
        nvtxRangePushA(name);
#endif
    }
    ~NvtxRange() {
#ifdef USE_NVTX
        nvtxRangePop();
#endif
    }
};

/// Macro for easy timing + NVTX annotation
/// Uses __COUNTER__ for unique variable names (nvc++ expands __LINE__ literally in ## context)
#define TIMED_SCOPE_CONCAT2(a, b) a##b
#define TIMED_SCOPE_CONCAT(a, b) TIMED_SCOPE_CONCAT2(a, b)
#define TIMED_SCOPE(name) nncfd::AutoTimer TIMED_SCOPE_CONCAT(_timer_, __COUNTER__)(name)

/// NVTX-only scoped range (no timing overhead, visible in nsys)
#define NVTX_SCOPE(name) nncfd::NvtxRange TIMED_SCOPE_CONCAT(_nvtx_, __COUNTER__)(name)

/// NVTX instant marker (zero-duration event visible in nsys timeline)
#ifdef USE_NVTX
#define NVTX_MARK(name) nvtxMarkA(name)
#else
#define NVTX_MARK(name) ((void)0)
#endif

} // namespace nncfd


