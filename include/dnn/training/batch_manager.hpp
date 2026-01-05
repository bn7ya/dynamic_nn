#pragma once

#include "../core/tensor.hpp"
#include "../core/random.hpp"
#include <vector>
#include <cmath>
#include <algorithm>

namespace dnn {
namespace training {

using core::Tensor;
using core::Random;

/**
 * Configuration for adaptive batch sizing.
 */
struct BatchConfig {
    size_t min_batch_size = 4;          // Start with small batches (overfit)
    size_t max_batch_size = 256;        // End with large batches (generalize)
    double growth_rate = 1.5;           // Multiplicative growth per interval
    size_t growth_interval_epochs = 5;  // Epochs between growth
    bool shuffle = true;                // Shuffle data each epoch
    bool power_of_two = true;           // Round batch size to power of 2 (better memory alignment)
};

/**
 * Iterator for batches.
 */
template<typename T>
class BatchIterator {
public:
    BatchIterator(const std::vector<Tensor<T>>& data,
                  const std::vector<Tensor<T>>& labels,
                  const std::vector<size_t>& indices,
                  size_t batch_size)
        : data_(data)
        , labels_(labels)
        , indices_(indices)
        , batch_size_(batch_size)
        , current_idx_(0) {}

    /**
     * Check if more batches available.
     */
    bool has_next() const {
        return current_idx_ < indices_.size();
    }

    /**
     * Get next batch.
     */
    std::pair<std::vector<Tensor<T>>, std::vector<Tensor<T>>> next() {
        size_t end_idx = std::min(current_idx_ + batch_size_, indices_.size());
        size_t actual_batch_size = end_idx - current_idx_;

        std::vector<Tensor<T>> batch_data;
        std::vector<Tensor<T>> batch_labels;
        batch_data.reserve(actual_batch_size);
        batch_labels.reserve(actual_batch_size);

        for (size_t i = current_idx_; i < end_idx; ++i) {
            size_t idx = indices_[i];
            batch_data.push_back(data_[idx]);
            batch_labels.push_back(labels_[idx]);
        }

        current_idx_ = end_idx;
        return {batch_data, batch_labels};
    }

    /**
     * Reset iterator.
     */
    void reset() {
        current_idx_ = 0;
    }

    /**
     * Get number of batches.
     */
    size_t num_batches() const {
        return (indices_.size() + batch_size_ - 1) / batch_size_;
    }

    /**
     * Get current batch index.
     */
    size_t current_batch() const {
        return current_idx_ / batch_size_;
    }

private:
    const std::vector<Tensor<T>>& data_;
    const std::vector<Tensor<T>>& labels_;
    std::vector<size_t> indices_;
    size_t batch_size_;
    size_t current_idx_;
};

/**
 * Manages adaptive batch sizing during training.
 * Starts with small batches (encourages overfitting/memorization),
 * grows to large batches (encourages generalization).
 */
template<typename T = float>
class BatchManager {
public:
    explicit BatchManager(const BatchConfig& config = BatchConfig())
        : config_(config)
        , current_batch_size_(config.min_batch_size)
        , current_epoch_(0)
        , rng_(42) {

        if (config_.power_of_two) {
            current_batch_size_ = next_power_of_two(current_batch_size_);
        }
    }

    /**
     * Create iterator for current epoch.
     */
    BatchIterator<T> create_iterator(const std::vector<Tensor<T>>& data,
                                     const std::vector<Tensor<T>>& labels) {
        std::vector<size_t> indices(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            indices[i] = i;
        }

        if (config_.shuffle) {
            rng_.shuffle(indices);
        }

        return BatchIterator<T>(data, labels, indices, current_batch_size_);
    }

    /**
     * Update batch size for new epoch.
     */
    void update_epoch(uint64_t epoch) {
        current_epoch_ = epoch;

        // Compute batch size based on epoch
        size_t intervals_passed = epoch / config_.growth_interval_epochs;
        double multiplier = std::pow(config_.growth_rate, static_cast<double>(intervals_passed));
        size_t new_size = static_cast<size_t>(config_.min_batch_size * multiplier);

        // Clamp to bounds
        new_size = std::max(config_.min_batch_size, std::min(config_.max_batch_size, new_size));

        if (config_.power_of_two) {
            new_size = next_power_of_two(new_size);
            new_size = std::min(new_size, config_.max_batch_size);
        }

        current_batch_size_ = new_size;
    }

    /**
     * Adjust batch size based on efficiency.
     * Low efficiency -> increase batch size faster (need more generalization)
     */
    void adjust_for_efficiency(double efficiency_score) {
        if (efficiency_score < 0.3) {
            // Very low efficiency - jump to larger batches
            size_t new_size = static_cast<size_t>(current_batch_size_ * 2.0);
            new_size = std::min(new_size, config_.max_batch_size);
            if (config_.power_of_two) {
                new_size = next_power_of_two(new_size);
                new_size = std::min(new_size, config_.max_batch_size);
            }
            current_batch_size_ = new_size;
        } else if (efficiency_score < 0.5) {
            // Low efficiency - moderate increase
            size_t new_size = static_cast<size_t>(current_batch_size_ * 1.5);
            new_size = std::min(new_size, config_.max_batch_size);
            if (config_.power_of_two) {
                new_size = next_power_of_two(new_size);
                new_size = std::min(new_size, config_.max_batch_size);
            }
            current_batch_size_ = new_size;
        }
        // High efficiency - keep normal schedule
    }

    /**
     * Get current batch size.
     */
    size_t current_batch_size() const {
        return current_batch_size_;
    }

    /**
     * Get configuration.
     */
    const BatchConfig& config() const {
        return config_;
    }

    /**
     * Set random seed.
     */
    void set_seed(uint64_t seed) {
        rng_.reseed(seed);
    }

    /**
     * Compute coverage ratio (how much of max batch size we're using).
     */
    double coverage_ratio() const {
        return static_cast<double>(current_batch_size_) / config_.max_batch_size;
    }

    /**
     * Check if at maximum batch size.
     */
    bool at_max_batch_size() const {
        return current_batch_size_ >= config_.max_batch_size;
    }

private:
    static size_t next_power_of_two(size_t n) {
        if (n == 0) return 1;
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    BatchConfig config_;
    size_t current_batch_size_;
    uint64_t current_epoch_;
    Random rng_;
};

/**
 * Dataset wrapper for training.
 */
template<typename T = float>
class Dataset {
public:
    Dataset(std::vector<Tensor<T>> inputs, std::vector<Tensor<T>> targets)
        : inputs_(std::move(inputs))
        , targets_(std::move(targets)) {

        if (inputs_.size() != targets_.size()) {
            throw std::invalid_argument("Inputs and targets must have same size");
        }
    }

    size_t size() const { return inputs_.size(); }

    const std::vector<Tensor<T>>& inputs() const { return inputs_; }
    const std::vector<Tensor<T>>& targets() const { return targets_; }

    /**
     * Split dataset into train/validation/test.
     * @param train_ratio Fraction for training (e.g., 0.8)
     * @param val_ratio Fraction for validation (e.g., 0.1)
     * @return {train, validation, test} datasets
     */
    std::tuple<Dataset, Dataset, Dataset> split(double train_ratio = 0.8,
                                                 double val_ratio = 0.1,
                                                 uint64_t seed = 42) const {
        Random rng(seed);
        auto indices = rng.permutation(inputs_.size());

        size_t train_end = static_cast<size_t>(inputs_.size() * train_ratio);
        size_t val_end = train_end + static_cast<size_t>(inputs_.size() * val_ratio);

        std::vector<Tensor<T>> train_inputs, train_targets;
        std::vector<Tensor<T>> val_inputs, val_targets;
        std::vector<Tensor<T>> test_inputs, test_targets;

        for (size_t i = 0; i < indices.size(); ++i) {
            size_t idx = indices[i];
            if (i < train_end) {
                train_inputs.push_back(inputs_[idx]);
                train_targets.push_back(targets_[idx]);
            } else if (i < val_end) {
                val_inputs.push_back(inputs_[idx]);
                val_targets.push_back(targets_[idx]);
            } else {
                test_inputs.push_back(inputs_[idx]);
                test_targets.push_back(targets_[idx]);
            }
        }

        return {
            Dataset(std::move(train_inputs), std::move(train_targets)),
            Dataset(std::move(val_inputs), std::move(val_targets)),
            Dataset(std::move(test_inputs), std::move(test_targets))
        };
    }

private:
    std::vector<Tensor<T>> inputs_;
    std::vector<Tensor<T>> targets_;
};

} // namespace training
} // namespace dnn
