#include "s3m/core/timer.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

namespace s3m {

double FpsMeter::tick() {
    const double dt = since_last_.elapsedSec();
    since_last_.reset();
    if (dt <= 0.0)
        return fps_;
    const double instantaneous = 1.0 / dt;
    if (first_) {
        first_ = false;
        fps_ = instantaneous;
    } else {
        fps_ = smoothing_ * fps_ + (1.0 - smoothing_) * instantaneous;
    }
    return fps_;
}

void ProfileRegistry::add(const std::string& name, double ms) {
    Entry& e = entries_[name];
    e.total_ms += ms;
    e.calls += 1;
}

double ProfileRegistry::totalMs(const std::string& name) const {
    const auto it = entries_.find(name);
    return it == entries_.end() ? 0.0 : it->second.total_ms;
}

std::int64_t ProfileRegistry::count(const std::string& name) const {
    const auto it = entries_.find(name);
    return it == entries_.end() ? 0 : it->second.calls;
}

double ProfileRegistry::meanMs(const std::string& name) const {
    const auto it = entries_.find(name);
    if (it == entries_.end() || it->second.calls == 0)
        return 0.0;
    return it->second.total_ms / static_cast<double>(it->second.calls);
}

std::string ProfileRegistry::summary() const {
    std::vector<std::pair<std::string, Entry>> rows(entries_.begin(), entries_.end());
    std::sort(rows.begin(), rows.end(),
              [](const std::pair<std::string, Entry>& a, const std::pair<std::string, Entry>& b) {
                  return a.second.total_ms > b.second.total_ms;
              });

    std::ostringstream os;
    os << std::left << std::setw(18) << "section" << std::right << std::setw(10) << "calls"
       << std::setw(14) << "total(ms)" << std::setw(14) << "mean(ms)" << '\n';
    os << std::string(56, '-') << '\n';
    os << std::fixed << std::setprecision(3);
    for (const auto& row : rows) {
        const Entry& e = row.second;
        const double mean = e.calls ? e.total_ms / static_cast<double>(e.calls) : 0.0;
        os << std::left << std::setw(18) << row.first << std::right << std::setw(10) << e.calls
           << std::setw(14) << e.total_ms << std::setw(14) << mean << '\n';
    }
    return os.str();
}

}  // namespace s3m
