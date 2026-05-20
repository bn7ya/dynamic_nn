#pragma once

#include <cstddef>
#include <utility>
#include <vector>


namespace enn::modules {
class ReversibleNetworkImpl;
}


namespace enn::controllers {


enum class TrainableSchedule { Linear, Exponential, Cosine, Step };
enum class SelectionStrategy {
    LeastUtilized,
    MostUtilized,
    Random,
    Distributed,
};


struct TrainableConfig {
    double initial_fraction = 1.0;
    double final_fraction = 0.5;
    std::size_t transition_epochs = 100;
    TrainableSchedule schedule = TrainableSchedule::Exponential;
    std::size_t step_count = 5;
    double exponential_decay_k = 5.0;
    SelectionStrategy strategy = SelectionStrategy::LeastUtilized;
};


class TrainableScheduler {
public:
    TrainableScheduler(enn::modules::ReversibleNetworkImpl& net,
                       TrainableConfig cfg = {});

    void update_epoch(std::size_t epoch);
    double current_trainable_fraction() const noexcept { return current_; }
    std::vector<std::pair<std::size_t, double>>
    schedule_curve(std::size_t num_points = 100) const;

    const TrainableConfig& config() const noexcept { return config_; }

private:
    double compute_fraction_for_epoch(std::size_t epoch) const;
    void apply_trainability();

    enn::modules::ReversibleNetworkImpl& network_;
    TrainableConfig config_;
    double current_;
    std::size_t current_epoch_ = 0;
};


}  // namespace enn::controllers
