#include "enn/controllers/trainable_scheduler.hpp"

#include <algorithm>
#include <cmath>


namespace enn::controllers {


TrainableScheduler::TrainableScheduler(
    enn::modules::ReversibleNetworkImpl& net, TrainableConfig cfg)
    : network_(net), config_(cfg), current_(cfg.initial_fraction) {}


void TrainableScheduler::update_epoch(std::size_t epoch) {
    current_epoch_ = epoch;
    current_ = compute_fraction_for_epoch(epoch);
    apply_trainability();
}


double TrainableScheduler::compute_fraction_for_epoch(
    std::size_t epoch) const {
    if (config_.transition_epochs == 0) return config_.final_fraction;
    double progress = std::min(1.0, static_cast<double>(epoch) /
                                       static_cast<double>(
                                           config_.transition_epochs));
    double initial = config_.initial_fraction;
    double final_v = config_.final_fraction;
    switch (config_.schedule) {
        case TrainableSchedule::Linear:
            return initial + (final_v - initial) * progress;
        case TrainableSchedule::Exponential: {
            double k = config_.exponential_decay_k;
            return final_v +
                   (initial - final_v) * std::exp(-k * progress);
        }
        case TrainableSchedule::Cosine:
            return final_v + (initial - final_v) * 0.5 *
                                  (1.0 + std::cos(M_PI * progress));
        case TrainableSchedule::Step: {
            std::size_t steps = std::max<std::size_t>(1, config_.step_count);
            std::size_t step =
                std::min(steps, static_cast<std::size_t>(progress *
                                                          static_cast<double>(
                                                              steps)));
            double frac = static_cast<double>(step) /
                          static_cast<double>(steps);
            return initial + (final_v - initial) * frac;
        }
    }
    return final_v;
}


void TrainableScheduler::apply_trainability() {
    (void)network_;
}


std::vector<std::pair<std::size_t, double>>
TrainableScheduler::schedule_curve(std::size_t num_points) const {
    std::vector<std::pair<std::size_t, double>> out;
    if (num_points == 0) return out;
    for (std::size_t i = 0; i < num_points; ++i) {
        std::size_t epoch = (i * config_.transition_epochs) /
                             std::max<std::size_t>(1, num_points - 1);
        out.emplace_back(epoch, compute_fraction_for_epoch(epoch));
    }
    return out;
}


}  // namespace enn::controllers
