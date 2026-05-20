#include "enn/modules/reversible_network.hpp"

#include <algorithm>


namespace enn::modules {


ReversibleNetworkImpl::ReversibleNetworkImpl(ReversibleNetworkConfig cfg)
    : hidden_activation_(cfg.hidden_activation),
      capture_metrics_(cfg.capture_metrics),
      config_(cfg) {
    std::int64_t in_features = cfg.input_features;
    for (std::int64_t h : cfg.hidden_sizes) {
        layer_vec_.emplace_back(ReversibleLinear(in_features, h));
        layer_active_.push_back(true);
        in_features = h;
    }
    layer_vec_.emplace_back(ReversibleLinear(in_features,
                                               cfg.output_features));
    layer_active_.push_back(true);
    rebuild_registrations_();
}


void ReversibleNetworkImpl::rebuild_registrations_() {
    auto names = std::vector<std::string>{};
    for (auto& kv : named_children()) names.push_back(kv.key());
    for (auto& n : names) {
        replace_module(n, std::make_shared<torch::nn::Module>());
    }
    for (std::size_t i = 0; i < layer_vec_.size(); ++i) {
        std::string n = "layer_" + std::to_string(i);
        if (named_children().contains(n)) {
            replace_module(n, layer_vec_[i].ptr());
        } else {
            register_module(n, layer_vec_[i]);
        }
    }
}


ReversibleLinearImpl& ReversibleNetworkImpl::layer(std::size_t i) {
    return *layer_vec_[i];
}


const ReversibleLinearImpl& ReversibleNetworkImpl::layer(std::size_t i) const {
    return *layer_vec_[i];
}


torch::Tensor ReversibleNetworkImpl::apply_hidden(torch::Tensor x) const {
    switch (hidden_activation_) {
        case HiddenActivation::ReLU:
            return torch::nn::functional::relu(x);
        case HiddenActivation::GELU:
            return torch::nn::functional::gelu(x);
        case HiddenActivation::Tanh:
            return torch::tanh(x);
        case HiddenActivation::Identity:
            return x;
    }
    return x;
}


torch::Tensor ReversibleNetworkImpl::forward(torch::Tensor input) {
    torch::Tensor x = input;
    if (capture_metrics_) {
        cached_layer_inputs_.assign(layer_vec_.size(), torch::Tensor{});
    }
    for (std::size_t i = 0; i < layer_vec_.size(); ++i) {
        if (!layer_active_[i]) continue;
        if (capture_metrics_) {
            cached_layer_inputs_[i] = x.detach();
        }
        auto& l = layer(i);
        auto pre = l.forward(x);
        bool is_last = (i + 1 == layer_vec_.size());
        x = is_last ? pre : apply_hidden(pre);
    }
    return x;
}


std::size_t ReversibleNetworkImpl::num_active_layers() const {
    std::size_t c = 0;
    for (bool a : layer_active_) if (a) ++c;
    return c;
}


std::int64_t ReversibleNetworkImpl::num_hidden_nodes() const {
    std::int64_t total = 0;
    for (std::size_t i = 0; i + 1 < layer_vec_.size(); ++i) {
        if (layer_active_[i]) total += layer(i).active_count();
    }
    return total;
}


void ReversibleNetworkImpl::rebuild_layer_preserving_(std::size_t idx,
                                                       std::int64_t new_in) {
    if (idx >= layer_vec_.size()) return;
    auto& old = layer(idx);
    std::int64_t old_in = old.in_features();
    std::int64_t old_out = old.weight.size(0);
    if (old_in == new_in) return;
    auto opts = old.weight.options();
    auto new_w = torch::zeros({old_out, new_in}, opts);
    std::int64_t copy_in = std::min(old_in, new_in);
    new_w.index_put_({torch::indexing::Slice(),
                        torch::indexing::Slice(0, copy_in)},
                      old.weight.detach().index({torch::indexing::Slice(),
                                                  torch::indexing::Slice(
                                                      0, copy_in)}));
    auto fresh = ReversibleLinear(new_in, old_out);
    fresh->reinit_parameter("weight", new_w, true);
    fresh->reinit_parameter("bias", old.bias.detach().clone(), true);
    fresh->active_mask.set_data(old.active_mask.detach().clone());
    layer_vec_[idx] = fresh;
}


void ReversibleNetworkImpl::insert_layer(std::int64_t index,
                                           std::int64_t num_nodes) {
    if (index < 0 || index > static_cast<std::int64_t>(layer_vec_.size())) {
        return;
    }
    std::int64_t in_features = (index == 0)
        ? config_.input_features
        : layer(static_cast<std::size_t>(index - 1)).weight.size(0);
    auto inserted = ReversibleLinear(in_features, num_nodes);
    layer_vec_.insert(layer_vec_.begin() + index, inserted);
    layer_active_.insert(layer_active_.begin() + index, true);
    if (static_cast<std::size_t>(index + 1) < layer_vec_.size()) {
        rebuild_layer_preserving_(static_cast<std::size_t>(index + 1),
                                    num_nodes);
    }
    rebuild_registrations_();
    ++topology_version_;
}


void ReversibleNetworkImpl::remove_layer(std::int64_t index) {
    if (index < 0 ||
        index >= static_cast<std::int64_t>(layer_vec_.size()) - 1) {
        return;
    }
    layer_active_[static_cast<std::size_t>(index)] = false;
    if (static_cast<std::size_t>(index + 1) < layer_vec_.size()) {
        std::int64_t new_in = (index == 0)
            ? config_.input_features
            : layer(static_cast<std::size_t>(index - 1)).weight.size(0);
        rebuild_layer_preserving_(static_cast<std::size_t>(index + 1),
                                    new_in);
    }
    rebuild_registrations_();
    ++topology_version_;
}


std::int64_t ReversibleNetworkImpl::compact() {
    std::int64_t total = 0;
    for (std::size_t i = 0; i < layer_vec_.size(); ++i) {
        if (!layer_active_[i]) continue;
        total += layer(i).compact();
    }
    std::vector<ReversibleLinear> kept;
    std::vector<bool> kept_active;
    for (std::size_t i = 0; i < layer_vec_.size(); ++i) {
        if (layer_active_[i]) {
            kept.push_back(layer_vec_[i]);
            kept_active.push_back(true);
        }
    }
    layer_vec_ = std::move(kept);
    layer_active_ = std::move(kept_active);
    for (std::size_t i = 1; i < layer_vec_.size(); ++i) {
        std::int64_t new_in = layer(i - 1).weight.size(0);
        if (layer(i).in_features() != new_in) {
            rebuild_layer_preserving_(i, new_in);
        }
    }
    rebuild_registrations_();
    ++topology_version_;
    return total;
}


}  // namespace enn::modules
