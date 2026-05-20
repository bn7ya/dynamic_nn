#include "enn/training/phase_controller.hpp"

#include <algorithm>
#include <cmath>


namespace enn::training {


namespace {

constexpr std::size_t kMaxHistory = 1024;

void push_capped(std::vector<double>& v, double x) {
    v.push_back(x);
    if (v.size() > kMaxHistory) v.erase(v.begin());
}

void push_capped(std::vector<std::string>& v, std::string x) {
    v.push_back(std::move(x));
    if (v.size() > kMaxHistory) v.erase(v.begin());
}

torch::Tensor as_class_indices(const torch::Tensor& y) {
    if (y.dim() == 1) return y.to(torch::kInt64);
    return y.argmax(1).to(torch::kInt64);
}

}  // namespace


PhaseController::PhaseController(enn::modules::ReversibleNetworkImpl& net,
                                   CostFunction cost)
    : net_(net), cost_function_(cost) {}


torch::Tensor PhaseController::compute_loss(torch::Tensor pred,
                                              torch::Tensor target) {
    switch (cost_function_) {
        case CostFunction::MSE:
            return torch::nn::functional::mse_loss(pred, target);
        case CostFunction::CrossEntropy: {
            auto idx = as_class_indices(target);
            return torch::nn::functional::cross_entropy(pred, idx);
        }
    }
    return torch::zeros({});
}


double PhaseController::compute_global_utilization() {
    if (net_.num_layers() == 0) return 0.0;
    double acc = 0.0;
    std::size_t n = 0;
    for (std::size_t i = 0; i + 1 < net_.num_layers(); ++i) {
        if (!net_.layer_active(i)) continue;
        auto util = net_.layer(i).metrics.utilization_per_node();
        acc += util.mean().item<double>();
        ++n;
    }
    return n > 0 ? acc / static_cast<double>(n) : 0.0;
}


void PhaseController::log_epoch(double cost) {
    double util = compute_global_utilization();
    push_capped(result_.cost_trajectory, cost);
    push_capped(result_.utilization_trajectory, util);
    if (stability_) {
        stability_->update_topology(net_.num_active_layers(),
                                     net_.num_hidden_nodes());
        push_capped(result_.stability_state_trajectory,
                     enn::controllers::stability_state_name(
                         stability_->current_state()));
    }
    ++result_.epochs_completed;
}


void PhaseController::freeze_topology() {
    for (std::size_t i = 0; i < net_.num_layers(); ++i) {
        auto& l = net_.layer(i);
        l.weight.set_requires_grad(true);
        l.bias.set_requires_grad(true);
    }
}


TrainingResult PhaseController::fit(const torch::Tensor& X,
                                      const torch::Tensor& y) {
    stats_ = compute_dataset_statistics(X, y);
    schedule_ = derive_phase_schedule(stats_);
    pruning_ = derive_pruning_config(stats_);
    lr_cfg_ = derive_adaptive_lr_config(stats_);
    stability_cfg_ = derive_stability_config(stats_);
    plateau_cfg_ = derive_plateau_config(stats_);
    enn::controllers::LayerManagerConfig lmc;
    lmc.prune_requires_plateau = true;
    lmc.utilization_threshold = pruning_.prune_utilization_threshold;
    lmc.saturation_threshold = pruning_.growth_utilization_threshold;
    lmc.cka_redundancy_threshold = pruning_.cka_redundancy_threshold;
    stability_ = std::make_unique<enn::controllers::StabilityMonitor>(
        net_.num_active_layers(), net_.num_hidden_nodes(), stability_cfg_);
    layer_manager_ = std::make_unique<enn::controllers::LayerManager>(
        net_, *stability_, lmc);
    plateau_ =
        std::make_unique<enn::controllers::PlateauDetector>(plateau_cfg_);
    current_lr_ = schedule_.topology_discovery_lr;
    initialized_ = true;

    run_topology_discovery(X, y);
    run_convergence_rate_estimation(X, y);
    run_adaptive_training(X, y);
    run_frozen_architecture_finetuning(X, y);

    net_.compact();
    result_.topology_version_final = net_.topology_version();
    std::int64_t pc = 0;
    for (auto& p : net_.parameters()) pc += p.numel();
    result_.parameter_count_final = pc;
    if (result_.stopping_reason.empty()) {
        result_.stopping_reason = "schedule_complete";
    }
    return result_;
}


static PhaseRunResult train_n_epochs(
    enn::modules::ReversibleNetworkImpl& net, const torch::Tensor& X,
    const torch::Tensor& y, double lr, std::int64_t epochs,
    PhaseController& pc, CostFunction cost,
    std::function<void(std::int64_t)> hook = nullptr) {
    PhaseRunResult r;
    torch::optim::Adam opt(net.parameters(),
                            torch::optim::AdamOptions(lr));
    auto target = y;
    for (std::int64_t e = 0; e < epochs; ++e) {
        opt.zero_grad();
        auto pred = net.forward(X);
        torch::Tensor loss;
        switch (cost) {
            case CostFunction::MSE:
                loss = torch::nn::functional::mse_loss(pred, target);
                break;
            case CostFunction::CrossEntropy:
                loss = torch::nn::functional::cross_entropy(
                    pred,
                    target.dim() == 1
                        ? target.to(torch::kInt64)
                        : target.argmax(1).to(torch::kInt64));
                break;
        }
        loss.backward();
        opt.step();
        for (std::size_t i = 0; i + 1 < net.num_layers(); ++i) {
            if (!net.layer_active(i)) continue;
            auto& l = net.layer(i);
            if (!l.weight.grad().defined()) continue;
            if (i >= net.cached_layer_inputs_.size()) continue;
            auto& cached = net.cached_layer_inputs_[i];
            if (!cached.defined()) continue;
            auto pre =
                torch::nn::functional::linear(cached, l.weight, l.bias);
            auto grad_act = pre.mean(0).abs();
            l.update_node_metrics(pre, grad_act.unsqueeze(0));
        }
        r.final_cost = loss.item<double>();
        pc.log_epoch(r.final_cost);
        if (hook) hook(e);
        ++r.epochs_run;
    }
    r.final_utilization = pc.result().utilization_trajectory.empty()
                              ? 0.0
                              : pc.result().utilization_trajectory.back();
    return r;
}


PhaseRunResult PhaseController::run_topology_discovery(
    const torch::Tensor& X, const torch::Tensor& y) {
    auto hook = [&](std::int64_t) {
        double util = compute_global_utilization();
        plateau_->tick(util);
        if (layer_manager_) {
            layer_manager_->tick_utilization(util);
            if (plateau_->is_at_local_max()) {
                layer_manager_->auto_adjust();
                plateau_->notify_event_fired();
            }
        }
    };
    return train_n_epochs(net_, X, y, schedule_.topology_discovery_lr,
                            schedule_.topology_discovery_epochs, *this,
                            cost_function_, hook);
}


PhaseRunResult PhaseController::run_convergence_rate_estimation(
    const torch::Tensor& X, const torch::Tensor& y) {
    return train_n_epochs(net_, X, y, schedule_.convergence_estimation_lr,
                            schedule_.convergence_estimation_epochs, *this,
                            cost_function_);
}


PhaseRunResult PhaseController::run_adaptive_training(
    const torch::Tensor& X, const torch::Tensor& y) {
    auto hook = [&](std::int64_t) {
        auto [new_lr, action] =
            enn::controllers::step_adaptive_lr_controller(
                current_lr_, result_.cost_trajectory,
                result_.utilization_trajectory, lr_cfg_, lr_state_);
        current_lr_ = new_lr;
        (void)action;
    };
    return train_n_epochs(net_, X, y, schedule_.adaptive_training_lr_initial,
                            schedule_.adaptive_training_epochs, *this,
                            cost_function_, hook);
}


PhaseRunResult PhaseController::run_frozen_architecture_finetuning(
    const torch::Tensor& X, const torch::Tensor& y) {
    freeze_topology();
    return train_n_epochs(net_, X, y, schedule_.frozen_finetune_lr,
                            schedule_.frozen_finetune_epochs, *this,
                            cost_function_);
}


}  // namespace enn::training
