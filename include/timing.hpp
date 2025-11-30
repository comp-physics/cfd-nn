#pragma once

#include <chrono>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

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
    
private:
    TimingStats() = default;
    
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

/// Macro for easy timing
#define TIMED_SCOPE(name) nncfd::AutoTimer _timer_##__LINE__(name)

} // namespace nncfd


