#pragma once

#include <torch/torch.h>

#include <cstddef>
#include <vector>


namespace enn::training {


struct BatchConfig {
    std::size_t min_batch_size = 4;
    std::size_t max_batch_size = 256;
    double growth_rate = 1.5;
    std::size_t growth_interval_epochs = 5;
    bool shuffle = true;
    bool power_of_two = true;
    double fast_growth_utilization = 0.3;
    double fast_growth_factor = 2.0;
    double slow_growth_utilization = 0.5;
    double slow_growth_factor = 1.5;
};


class BatchManager {
public:
    BatchManager(std::size_t num_samples, BatchConfig cfg);

    std::vector<std::vector<std::int64_t>> shuffled_batches(
        std::int64_t seed) const;

    void update_epoch(std::size_t epoch, double utilization);
    std::size_t current_batch_size() const noexcept { return current_; }
    const BatchConfig& config() const noexcept { return config_; }

private:
    std::size_t round_to_pow2(std::size_t v) const;

    std::size_t num_samples_;
    BatchConfig config_;
    std::size_t current_;
    std::size_t epoch_ = 0;
};


}  // namespace enn::training
