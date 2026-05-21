#include "enn/modules/adaptive_conv2d.hpp"

#include <algorithm>


namespace enn::modules {


AdaptiveConv2dImpl::AdaptiveConv2dImpl(
    std::int64_t in_channels, std::int64_t out_channels,
    std::vector<std::int64_t> candidate_kernel_sizes)
    : kernel_sizes_(std::move(candidate_kernel_sizes)),
      in_channels_(in_channels),
      out_channels_(out_channels) {
    candidates_ = register_module("candidates", torch::nn::ModuleList());
    for (std::int64_t k : kernel_sizes_) {
        auto opts = torch::nn::Conv2dOptions(in_channels, out_channels, k)
                        .padding(k / 2);
        candidates_->push_back(torch::nn::Conv2d(opts));
    }
    architecture_logits_ = register_parameter(
        "architecture_logits",
        torch::zeros({static_cast<std::int64_t>(kernel_sizes_.size())}));
    candidate_mask_ = register_buffer(
        "candidate_mask",
        torch::ones({static_cast<std::int64_t>(kernel_sizes_.size())}));
    active_kernel_size_ = kernel_sizes_.front();
}


torch::Tensor AdaptiveConv2dImpl::softmax_with_mask() const {
    auto neg_inf = std::numeric_limits<float>::lowest();
    auto masked = torch::where(candidate_mask_ > 0,
                                architecture_logits_,
                                torch::full_like(architecture_logits_,
                                                  neg_inf));
    return torch::softmax(masked, 0);
}


torch::Tensor AdaptiveConv2dImpl::forward(torch::Tensor input) {
    auto weights = softmax_with_mask();
    torch::Tensor out;
    bool first = true;
    for (std::size_t i = 0; i < candidates_->size(); ++i) {
        auto& conv = candidates_->at<torch::nn::Conv2dImpl>(i);
        auto c = conv.forward(input) * weights[i];
        if (first) {
            out = c;
            first = false;
        } else {
            out = out + c;
        }
    }
    return out;
}


void AdaptiveConv2dImpl::prune_smallest_candidate() {
    auto masked_logits = architecture_logits_.detach() +
                          (candidate_mask_ - 1.0) * 1e9;
    auto idx = masked_logits.argmin().item<std::int64_t>();
    if (candidate_mask_.sum().item<double>() <= 1.0) return;
    candidate_mask_.index_put_({idx}, 0);
    auto& conv = candidates_->at<torch::nn::Conv2dImpl>(
        static_cast<std::size_t>(idx));
    for (auto& p : conv.parameters()) {
        p.set_requires_grad(false);
    }
    ++topology_version_;
}


void AdaptiveConv2dImpl::grow_active_kernel() {
    auto active = candidate_mask_.argmax().item<std::int64_t>();
    auto& conv = candidates_->at<torch::nn::Conv2dImpl>(
        static_cast<std::size_t>(active));
    auto& old_w = conv.weight;
    std::int64_t old_k = old_w.size(2);
    std::int64_t new_k = old_k + 2;
    auto opts = old_w.options();
    auto new_w = torch::zeros({out_channels_, in_channels_, new_k, new_k},
                               opts);
    new_w.index_put_(
        {torch::indexing::Slice(),
          torch::indexing::Slice(),
          torch::indexing::Slice(1, 1 + old_k),
          torch::indexing::Slice(1, 1 + old_k)},
        old_w.detach());
    conv.weight = conv.register_parameter("weight", new_w);
    conv.options.kernel_size({new_k, new_k}).padding({new_k / 2, new_k / 2});
    active_kernel_size_ = new_k;
    ++topology_version_;
}


}  // namespace enn::modules
