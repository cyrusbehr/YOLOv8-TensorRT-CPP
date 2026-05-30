#pragma once
// Minimal stopwatch for the ENABLE_BENCHMARKS timing and the benchmark executable. v6 got this
// from the engine library's util/Stopwatch.h, which v7 does not ship; this is a drop-in local
// replacement with the same preciseStopwatch interface.
#include <chrono>

template <typename Clock = std::chrono::steady_clock> class Stopwatch {
public:
    template <typename T, typename Duration> T elapsedTime() const {
        return static_cast<T>(std::chrono::duration_cast<Duration>(Clock::now() - start_).count());
    }

private:
    typename Clock::time_point start_ = Clock::now();
};

using preciseStopwatch = Stopwatch<std::chrono::high_resolution_clock>;
