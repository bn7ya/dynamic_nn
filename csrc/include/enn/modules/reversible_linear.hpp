#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <memory>

#include "enn/modules/node_metrics.hpp"


namespace enn::modules {


class ReversibleLinearImpl : public torch::nn::Module {
public:
    ReversibleLinearImpl(std::int64_t in_features, std::int64_t out_features);

    torch::Tensor forward(torch::Tensor input);

    std::int64_t add_nodes(std::int64_t n);
    void prune_nodes(const std::vector<std::int64_t>& indices);
    std::int64_t compact();
    void update_node_metrics(const torch::Tensor& pre_act,
                             const torch::Tensor& grad_act);

    std::int64_t in_features() const noexcept { return in_features_; }
    std::int64_t out_features() const noexcept { return weight.size(0); }
    std::int64_t active_count() const;
    std::int64_t topology_version() const noexcept {
        return topology_version_;
    }

    torch::Tensor weight;
    torch::Tensor bias;
    torch::Tensor active_mask;
    NodeMetrics metrics;

    void reinit_parameter(const std::string& name, torch::Tensor value,
                           bool requires_grad);

private:
    std::int64_t in_features_;
    std::int64_t topology_version_ = 0;
};

TORCH_MODULE(ReversibleLinear);


}  // namespace enn::modules
