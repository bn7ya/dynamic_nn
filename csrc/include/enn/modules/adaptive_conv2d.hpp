#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <vector>


namespace enn::modules {


class AdaptiveConv2dImpl : public torch::nn::Module {
public:
    AdaptiveConv2dImpl(std::int64_t in_channels, std::int64_t out_channels,
                       std::vector<std::int64_t> candidate_kernel_sizes);

    torch::Tensor forward(torch::Tensor input);

    void prune_smallest_candidate();
    void grow_active_kernel();

    std::int64_t active_kernel_size() const noexcept {
        return active_kernel_size_;
    }
    std::int64_t topology_version() const noexcept {
        return topology_version_;
    }
    const torch::Tensor& candidate_mask() const { return candidate_mask_; }
    const torch::Tensor& architecture_logits() const {
        return architecture_logits_;
    }

private:
    torch::Tensor softmax_with_mask() const;

    torch::nn::ModuleList candidates_{nullptr};
    torch::Tensor architecture_logits_;
    torch::Tensor candidate_mask_;
    std::vector<std::int64_t> kernel_sizes_;
    std::int64_t active_kernel_size_;
    std::int64_t in_channels_;
    std::int64_t out_channels_;
    std::int64_t topology_version_ = 0;
};

TORCH_MODULE(AdaptiveConv2d);


}  // namespace enn::modules
