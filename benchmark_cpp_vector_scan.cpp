// Native lower-bound benchmark for the cosine scan in build_synaptic_links.
//
// This is intentionally a standalone benchmark, not a production extension:
// it measures the arithmetic opportunity without hiding Python/C++ conversion
// costs. Compile with:
//   g++ -O3 -march=native -std=c++20 benchmark_cpp_vector_scan.cpp -o /tmp/ina-vector-bench
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const std::size_t count = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000;
    const std::size_t dims = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 64;
    const int repeats = argc > 3 ? std::max(1, std::atoi(argv[3])) : 3;

    std::mt19937 rng(static_cast<unsigned>(count));
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);
    std::vector<float> vectors(count * dims);
    for (std::size_t row = 0; row < count; ++row) {
        float squared_norm = 0.0F;
        for (std::size_t col = 0; col < dims; ++col) {
            const float value = distribution(rng);
            vectors[row * dims + col] = value;
            squared_norm += value * value;
        }
        const float inverse_norm = 1.0F / std::sqrt(std::max(squared_norm, 1.0e-12F));
        for (std::size_t col = 0; col < dims; ++col) {
            vectors[row * dims + col] *= inverse_norm;
        }
    }

    std::vector<double> timings;
    volatile std::size_t matches = 0;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto started = std::chrono::steady_clock::now();
        std::size_t local_matches = 0;
        for (std::size_t left = 0; left < count; ++left) {
            const float* a = vectors.data() + left * dims;
            for (std::size_t right = left + 1; right < count; ++right) {
                const float* b = vectors.data() + right * dims;
                float dot = 0.0F;
                for (std::size_t col = 0; col < dims; ++col) {
                    dot += a[col] * b[col];
                }
                local_matches += dot >= 1.1F;
            }
        }
        const auto stopped = std::chrono::steady_clock::now();
        matches = local_matches;
        timings.push_back(std::chrono::duration<double>(stopped - started).count());
    }
    std::sort(timings.begin(), timings.end());
    const double median = timings[timings.size() / 2];
    const std::size_t comparisons = count * (count - 1) / 2;
    std::cout << std::fixed << std::setprecision(6)
              << "{\"items\":" << count
              << ",\"dimensions\":" << dims
              << ",\"comparisons\":" << comparisons
              << ",\"median_seconds\":" << median
              << ",\"comparisons_per_second\":" << comparisons / median
              << ",\"matches\":" << matches << "}\n";
}
