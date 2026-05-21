#include "enn/modules/reversible_linear.hpp"

#include <algorithm>
#include <cmath>


namespace enn::modules {


namespace {

torch::Tensor he_init(std::int64_t out_features, std::int64_t in_features,
                      torch::TensorOptions opts) {
    double std_dev = std::sqrt(2.0 / static_cast<double>(in_features));
    return torch::randn({out_features, in_features}, opts) * std_dev;
}

}  // namespace


ReversibleLinearImpl::ReversibleLinearImpl(std::int64_t in_features,
                                            std::int64_t out_features)
    : metrics(out_features, NodeMetricsConfig{}),
      in_features_(in_features) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32);
    weight = register_parameter("weight",
                                 he_init(out_features, in_features, opts));
    bias = register_parameter("bias", torch::zeros({out_features}, opts));
    active_mask = register_buffer(
        "active_mask", torch::ones({out_features}, opts));
}


void ReversibleLinearImpl::reinit_parameter(const std::string& name,
                                             torch::Tensor value,
                                             bool requires_grad) {
    value.set_requires_grad(requires_grad);
    auto it = named_parameters().find(name);
    if (it != nullptr) {
        it->set_data(value);
        it->set_requires_grad(requires_grad);
        if (name == "weight") weight = *it;
        if (name == "bias") bias = *it;
    } else {
        auto p = register_parameter(name, value);
        if (name == "weight") weight = p;
        if (name == "bias") bias = p;
    }
}


torch::Tensor ReversibleLinearImpl::forward(torch::Tensor input) {
    auto out = torch::nn::functional::linear(input, weight, bias);
    return out * active_mask.to(out.dtype());
}


std::int64_t ReversibleLinearImpl::active_count() const {
    return active_mask.to(torch::kInt64).sum().item<std::int64_t>();
}


std::int64_t ReversibleLinearImpl::add_nodes(std::int64_t n) {
    if (n <= 0) return 0;
    auto dormant = (active_mask == 0).nonzero().squeeze(-1);
    std::int64_t reactivated = 0;
    if (dormant.numel() > 0) {
        auto take = std::min<std::int64_t>(n, dormant.numel());
        auto idx = dormant.index({torch::indexing::Slice(0, take)});
        active_mask.index_put_({idx}, 1);
        reactivated = take;
        n -= take;
    }
    if (n > 0) {
        auto opts = weight.options();
        auto new_w = he_init(n, in_features_, opts);
        auto new_b = torch::zeros({n}, opts);
        auto new_m = torch::ones({n}, opts);
        auto cat_w = torch::cat({weight.detach(), new_w}, 0);
        auto cat_b = torch::cat({bias.detach(), new_b}, 0);
        active_mask.set_data(torch::cat({active_mask, new_m}, 0));
        reinit_parameter("weight", cat_w, true);
        reinit_parameter("bias", cat_b, true);
        metrics.resize(weight.size(0));
    }
    ++topology_version_;
    return reactivated + n;
}


void ReversibleLinearImpl::prune_nodes(
    const std::vector<std::int64_t>& indices) {
    if (indices.empty()) return;
    auto idx = torch::tensor(indices, torch::TensorOptions().dtype(
                                          torch::kInt64));
    active_mask.index_put_({idx}, 0);
    ++topology_version_;
}


std::int64_t ReversibleLinearImpl::compact() {
    auto keep = (active_mask > 0).nonzero().squeeze(-1);
    std::int64_t dropped = weight.size(0) - keep.numel();
    ++topology_version_;
    if (dropped == 0) return 0;
    auto new_w = weight.detach().index({keep}).contiguous();
    auto new_b = bias.detach().index({keep}).contiguous();
    auto new_m = torch::ones({keep.numel()}, weight.options());
    active_mask.set_data(new_m);
    reinit_parameter("weight", new_w, true);
    reinit_parameter("bias", new_b, true);
    metrics.resize(weight.size(0));
    return dropped;
}


void ReversibleLinearImpl::update_node_metrics(const torch::Tensor& pre_act,
                                                const torch::Tensor& grad_act) {
    metrics.observe(pre_act, grad_act);
}


}  // namespace enn::modules
