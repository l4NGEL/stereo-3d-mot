#pragma once

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

namespace s3m {

/// Monotonic stopwatch based on std::chrono::steady_clock.
class Stopwatch {
 public:
    Stopwatch() : start_(Clock::now()) {}

    void reset() { start_ = Clock::now(); }

    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }
    double elapsedSec() const {
        return std::chrono::duration<double>(Clock::now() - start_).count();
    }

 private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

/// Exponential moving average of a processing rate (Hz).
class FpsMeter {
 public:
    explicit FpsMeter(double smoothing = 0.9) : smoothing_(smoothing) {}

    /// Call once per processed item. Returns the current smoothed rate.
    double tick();

    double fps() const { return fps_; }

 private:
    double smoothing_;
    double fps_ = 0.0;
    Stopwatch since_last_;
    bool first_ = true;
};

/// Accumulating profiler. Create RAII scopes with scope("name"); read a report
/// with summary().
class ProfileRegistry {
 public:
    /// Add an elapsed measurement to a named section.
    void add(const std::string& name, double ms);

    double totalMs(const std::string& name) const;
    std::int64_t count(const std::string& name) const;
    double meanMs(const std::string& name) const;

    /// Table ordered by total time, descending.
    std::string summary() const;

    /// RAII timer: measures its lifetime and adds it to `name` on destruction.
    class Scope {
     public:
        Scope(ProfileRegistry& registry, std::string name)
            : registry_(&registry), name_(std::move(name)) {}
        ~Scope() {
            if (registry_) registry_->add(name_, sw_.elapsedMs());
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        Scope(Scope&& other) noexcept
            : registry_(other.registry_), name_(std::move(other.name_)), sw_(other.sw_) {
            other.registry_ = nullptr;
        }
        Scope& operator=(Scope&&) = delete;

     private:
        ProfileRegistry* registry_;
        std::string name_;
        Stopwatch sw_;
    };

    Scope scope(const std::string& name) { return Scope(*this, name); }

 private:
    struct Entry {
        double total_ms = 0.0;
        std::int64_t calls = 0;
    };
    std::map<std::string, Entry> entries_;
};

}  // namespace s3m
