#include "enn/controllers/plateau_detector.hpp"

#include <cmath>


namespace enn::controllers {


PlateauDetector::PlateauDetector(PlateauConfig cfg) : config_(cfg) {
    if (config_.window < 3) {
        throw std::invalid_argument(
            "PlateauConfig.window must be >= 3 for finite second difference");
    }
}


void PlateauDetector::tick(double utilization) {
    samples_.push_back(utilization);
    if (samples_.size() > config_.window) {
        samples_.pop_front();
    }
    ++epochs_seen_;
    if (cooldown_remaining_ > 0) {
        --cooldown_remaining_;
    }
}


void PlateauDetector::notify_event_fired() {
    cooldown_remaining_ = config_.cooldown_epochs;
}


double PlateauDetector::slope() const {
    const std::size_t n = samples_.size();
    if (n < 2) return 0.0;
    double t_mean = 0.5 * static_cast<double>(n - 1);
    double y_mean = 0.0;
    for (double v : samples_) y_mean += v;
    y_mean /= static_cast<double>(n);
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double dt = static_cast<double>(i) - t_mean;
        num += dt * (samples_[i] - y_mean);
        den += dt * dt;
    }
    return den > 0.0 ? num / den : 0.0;
}


double PlateauDetector::curvature() const {
    const std::size_t n = samples_.size();
    if (n < 3) return 0.0;
    double acc = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 1; i + 1 < n; ++i) {
        acc += samples_[i + 1] - 2.0 * samples_[i] + samples_[i - 1];
        ++count;
    }
    return count > 0 ? acc / static_cast<double>(count) : 0.0;
}


bool PlateauDetector::is_at_local_max() const {
    if (samples_.size() < config_.window) return false;
    if (epochs_seen_ < config_.warmup_epochs) return false;
    if (cooldown_remaining_ > 0) return false;
    if (std::abs(slope()) > config_.slope_epsilon) return false;
    return curvature() <= 0.0;
}


}  // namespace enn::controllers
