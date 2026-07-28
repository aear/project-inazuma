#include <cmath>
#include <cstddef>
#include <cstdint>

extern "C" std::size_t inazuma_cosine_pairs(
    const double* vectors, std::size_t rows, std::size_t dimensions,
    double threshold, std::size_t pair_limit, std::size_t per_source_limit,
    std::uint32_t* left_out, std::uint32_t* right_out, double* score_out,
    std::size_t output_capacity, std::size_t* pairs_evaluated, int* truncated) {
    if (!vectors || !left_out || !right_out || !score_out || !pairs_evaluated || !truncated) return 0;
    *pairs_evaluated = 0;
    *truncated = 0;
    std::size_t emitted = 0;
    for (std::size_t left = 0; left < rows; ++left) {
        std::size_t source_edges = 0;
        const double* a = vectors + left * dimensions;
        for (std::size_t right = left + 1; right < rows; ++right) {
            if (per_source_limit && source_edges >= per_source_limit) break;
            if (pair_limit && *pairs_evaluated >= pair_limit) {
                *truncated = 1;
                return emitted;
            }
            ++*pairs_evaluated;
            const double* b = vectors + right * dimensions;
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (std::size_t col = 0; col < dimensions; ++col) {
                const double av = a[col], bv = b[col];
                dot += av * bv;
                norm_a += av * av;
                norm_b += bv * bv;
            }
            const double similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1.0e-8);
            if (similarity < threshold) continue;
            if (emitted >= output_capacity) {
                *truncated = 1;
                return emitted;
            }
            left_out[emitted] = static_cast<std::uint32_t>(left);
            right_out[emitted] = static_cast<std::uint32_t>(right);
            score_out[emitted] = similarity;
            ++emitted;
            ++source_edges;
        }
    }
    return emitted;
}
