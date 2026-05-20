#include <gtest/gtest.h>

#include <vector>

#include "dnn/core/network.hpp"
#include "dnn/core/tensor.hpp"
#include "dnn/dynamics/health_monitor.hpp"
#include "dnn/dynamics/layer_manager.hpp"

using dnn::core::Network;
using dnn::core::NetworkConfig;
using dnn::core::Tensor;
using dnn::dynamics::HealthMonitor;
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

// Self-review fix: the probe must seed a fresh RNG per layer so two
// probes of the SAME layer use the same input batch. Previously the
// RNG was shared, advancing between calls; identical layers scored
// near 0 instead of near 1, exactly the failure mode the CKA fix was
// supposed to eliminate.
TEST(LayerSimilarity, SameLayerProbeYieldsHighSimilarity) {
    NetworkConfig nc;
    nc.input_shape = {6};
    nc.output_size = 3;
    nc.seed = 11;
    Network<float> net(nc);
    HealthMonitor<float> hm(net);
    LM lm(net, hm);

    double sim = lm.compute_layer_similarity_for_testing(0, 0);
    EXPECT_NEAR(sim, 1.0, 1e-4);
}
