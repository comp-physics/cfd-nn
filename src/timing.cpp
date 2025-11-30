#include "timing.hpp"

namespace nncfd {

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


