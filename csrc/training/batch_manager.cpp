#include "enn/training/batch_manager.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>


namespace enn::training {


BatchManager::BatchManager(std::size_t num_samples, BatchConfig cfg)
    : num_samples_(num_samples),
      config_(cfg),
      current_(std::max(cfg.min_batch_size, std::size_t{1})) {}


std::size_t BatchManager::round_to_pow2(std::size_t v) const {
    std::size_t p = 1;
    while (p < v) p <<= 1;
    if (p > v) {
        std::size_t lower = p >> 1;
        return (v - lower) < (p - v) ? lower : p;
    }
    return p;
}


std::vector<std::vector<std::int64_t>> BatchManager::shuffled_batches(
    std::int64_t seed) const {
    std::vector<std::int64_t> indices(num_samples_);
    std::iota(indices.begin(), indices.end(), 0);
    if (config_.shuffle) {
        std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
        std::shuffle(indices.begin(), indices.end(), rng);
    }
    std::vector<std::vector<std::int64_t>> out;
    for (std::size_t i = 0; i < indices.size(); i += current_) {
        auto end = std::min(indices.size(), i + current_);
        out.emplace_back(indices.begin() + i, indices.begin() + end);
    }
    return out;
}


void BatchManager::update_epoch(std::size_t epoch, double utilization) {
    epoch_ = epoch;
    double next = static_cast<double>(current_);
    if (utilization < config_.fast_growth_utilization) {
        next *= config_.fast_growth_factor;
    } else if (utilization < config_.slow_growth_utilization) {
        next *= config_.slow_growth_factor;
    } else if (epoch_ > 0 && epoch_ % config_.growth_interval_epochs == 0) {
        next *= config_.growth_rate;
    }
    std::size_t clamped = static_cast<std::size_t>(std::round(next));
    clamped = std::max(config_.min_batch_size,
                        std::min(config_.max_batch_size, clamped));
    if (config_.power_of_two) clamped = round_to_pow2(clamped);
    current_ = clamped;
}


}  // namespace enn::training
