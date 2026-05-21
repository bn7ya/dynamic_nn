#pragma once

#include <cstddef>
#include <deque>
#include <stdexcept>


namespace enn::controllers {


struct PlateauConfig {
    std::size_t window = 10;
    double slope_epsilon = 1e-4;
    std::size_t warmup_epochs = 8;
    std::size_t cooldown_epochs = 5;
};


class PlateauDetector {
public:
    PlateauDetector() : PlateauDetector(PlateauConfig{}) {}
    explicit PlateauDetector(PlateauConfig cfg);

    void tick(double utilization);
    void notify_event_fired();

    double slope() const;
    double curvature() const;
    bool is_at_local_max() const;

    std::size_t epochs_seen() const noexcept { return epochs_seen_; }
    std::size_t window_size() const noexcept { return samples_.size(); }
    const PlateauConfig& config() const noexcept { return config_; }

private:
    PlateauConfig config_;
    std::deque<double> samples_;
    std::size_t epochs_seen_ = 0;
    std::size_t cooldown_remaining_ = 0;
};


}  // namespace enn::controllers
