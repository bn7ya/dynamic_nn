#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dnn/core/layer.hpp"
#include "dnn/core/random.hpp"

using dnn::core::ActivationType;
using dnn::core::Layer;
using dnn::core::Random;

// 1.12: compute_gradient_threshold must ignore soft-removed nodes.
// Stamp small gradients on nodes 0..2 and huge gradients on 3..5, then
// soft-remove 3..5. The threshold must equal mean+std over the active
// nodes only -- the outliers must not drag it.
TEST(Layer, GradientThresholdIgnoresInactiveNodes) {
    Layer<float> layer(4, 6, ActivationType::ReLU, /*seed=*/1);

    for (int s = 0; s < 32; ++s) {
        for (size_t i = 0; i < 3; ++i) layer.node(i).record_gradient(0.01f);
        for (size_t i = 3; i < 6; ++i) layer.node(i).record_gradient(50.0f);
    }

    layer.remove_nodes({3, 4, 5});  // soft-remove the outliers

    // Expected: mean+std of gradient_magnitude_avg over the 3 active nodes.
    std::vector<double> g;
    for (size_t i = 0; i < 3; ++i)
        g.push_back(layer.node(i).compute_metrics().gradient_magnitude_avg);
    double mean = (g[0] + g[1] + g[2]) / 3.0;
    double sq = 0.0;
    for (double v : g) sq += (v - mean) * (v - mean);
    double expected = mean + std::sqrt(sq / 3.0);

    EXPECT_NEAR(layer.compute_gradient_threshold(), expected, 1e-5);
}

// 1.1: clip_gradients clamps every gradient element into [-clip, clip].
TEST(Layer, ClipGradientsClampsExtremeValues) {
    Layer<float> layer(5, 4, ActivationType::ReLU, /*seed=*/2);
    layer.zero_gradients();

    auto& wg = layer.weight_gradients();
    auto& bg = layer.bias_gradients();
    for (size_t k = 0; k < wg.size(); ++k)
        wg.data()[k] = (k % 2 == 0) ? 1e6f : -1e6f;
    for (size_t k = 0; k < bg.size(); ++k) bg.data()[k] = -42.0f;

    const float clip = 1.5f;
    layer.clip_gradients(clip);

    for (size_t k = 0; k < wg.size(); ++k) {
        EXPECT_LE(wg.data()[k], clip);
        EXPECT_GE(wg.data()[k], -clip);
    }
    for (size_t k = 0; k < bg.size(); ++k)
        EXPECT_NEAR(bg.data()[k], -clip, 1e-6);
}

// 1.2: perturb_node is seeded-deterministic and skips inactive nodes.
TEST(Layer, PerturbNodeDeterministicAndSkipsInactive) {
    auto build = []() {
        auto l = std::make_unique<Layer<float>>(4, 3, ActivationType::ReLU, 5);
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 4; ++j)
                l->weights().at(i, j) = static_cast<float>(i + j);
        return l;
    };

    auto a = build();
    auto b = build();
    a->remove_nodes({1});  // soft-remove node 1
    b->remove_nodes({1});

    Random ra(123), rb(123);
    for (size_t node = 0; node < 3; ++node) {
        a->perturb_node(node, 0.05f, ra);
        b->perturb_node(node, 0.05f, rb);
    }

    // Inactive node 1 untouched.
    for (size_t j = 0; j < 4; ++j)
        EXPECT_FLOAT_EQ(a->weights().at(1, j), static_cast<float>(1 + j));
    // Active nodes 0 and 2 changed.
    EXPECT_NE(a->weights().at(0, 0), 0.0f);
    EXPECT_NE(a->weights().at(2, 0), 2.0f);
    // Same seed -> identical perturbation.
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 4; ++j)
            EXPECT_FLOAT_EQ(a->weights().at(i, j), b->weights().at(i, j));
}
