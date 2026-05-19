#include <gtest/gtest.h>

#include <vector>

#include "dnn/core/tensor.hpp"
#include "dnn/dynamics/layer_manager.hpp"

using dnn::core::Tensor;
using LM = dnn::dynamics::LayerManager<float>;

// 3.4: identical representations -> CKA ~ 1; an independent random
// representation of the SAME shape -> clearly < 1 (the old size+eff
// heuristic would have called these "similar").
TEST(LayerSimilarity, LinearCKADiscriminatesFunction) {
    const size_t n = 16, d = 8;
    Tensor<float> A(std::vector<size_t>{n, d});
    Tensor<float> Acopy(std::vector<size_t>{n, d});
    Tensor<float> B(std::vector<size_t>{n, d});

    uint32_t s = 12345u;
    auto rnd = [&]() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFF) / 65535.0f - 0.5f;
    };
    for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < d; ++c) {
            float v = rnd();
            A.at(r, c) = v;
            Acopy.at(r, c) = v;
            B.at(r, c) = rnd();  // independent
        }

    double self_sim = LM::linear_cka(A, Acopy, n);
    double cross_sim = LM::linear_cka(A, B, n);

    EXPECT_NEAR(self_sim, 1.0, 1e-6);
    EXPECT_LT(cross_sim, 0.9);
    EXPECT_GE(cross_sim, 0.0);
}
