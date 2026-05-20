#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <vector>

#include "enn/modules/reversible_linear.hpp"


namespace enn::modules {


enum class HiddenActivation { ReLU, GELU, Tanh, Identity };


struct ReversibleNetworkConfig {
    std::int64_t input_features;
    std::int64_t output_features;
    std::vector<std::int64_t> hidden_sizes;
    HiddenActivation hidden_activation = HiddenActivation::ReLU;
    bool capture_metrics = true;
};


class ReversibleNetworkImpl : public torch::nn::Module {
public:
    explicit ReversibleNetworkImpl(ReversibleNetworkConfig cfg);

    torch::Tensor forward(torch::Tensor input);
    void insert_layer(std::int64_t index, std::int64_t num_nodes);
    void remove_layer(std::int64_t index);
    std::int64_t compact();

    std::size_t num_layers() const noexcept { return layer_vec_.size(); }
    std::size_t num_active_layers() const;
    std::int64_t num_hidden_nodes() const;
    std::int64_t topology_version() const noexcept {
        return topology_version_;
    }

    ReversibleLinearImpl& layer(std::size_t i);
    const ReversibleLinearImpl& layer(std::size_t i) const;
    bool layer_active(std::size_t i) const { return layer_active_[i]; }

    std::vector<ReversibleLinear> layer_vec_;
    std::vector<bool> layer_active_;
    std::vector<torch::Tensor> cached_layer_inputs_;
    HiddenActivation hidden_activation_;
    bool capture_metrics_ = true;

private:
    torch::Tensor apply_hidden(torch::Tensor x) const;
    void rebuild_layer_preserving_(std::size_t idx, std::int64_t new_in);
    void rebuild_registrations_();

    ReversibleNetworkConfig config_;
    std::int64_t topology_version_ = 0;
    std::size_t name_counter_ = 0;
};

TORCH_MODULE(ReversibleNetwork);


}  // namespace enn::modules
