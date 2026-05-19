#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "dnn/training/optimizer.hpp"
#include "dnn/training/trainer.hpp"

using dnn::training::AdamOptimizer;
using dnn::training::Optimizer;
using dnn::training::OptimizerConfig;
using dnn::training::OptimizerType;
using dnn::training::SGDOptimizer;
using dnn::training::TrainerConfig;
using dnn::core::Tensor;

namespace {
// Minimise f(w) = sum(w^2); grad = 2w. Run N steps, return |w|.
template <class Opt>
double final_norm(Opt& opt, double lr) {
    Tensor<float> w(std::vector<size_t>{8}, 1.0f);
    Tensor<float> b(std::vector<size_t>{1}, 0.0f);
    opt.initialize(w.size(), b.size());
    opt.set_learning_rate(static_cast<float>(lr));
    for (int step = 0; step < 200; ++step) {
        Tensor<float> dw(std::vector<size_t>{8});
        for (size_t i = 0; i < w.size(); ++i) dw.data()[i] = 2.0f * w.data()[i];
        Tensor<float> db(std::vector<size_t>{1}, 0.0f);
        opt.update(w, b, dw, db);
        opt.step();
    }
    double n = 0.0;
    for (size_t i = 0; i < w.size(); ++i) n += w.data()[i] * w.data()[i];
    return std::sqrt(n);
}
}  // namespace

// 1.3: Adam (now reachable through the trainer) actually optimises -
// it drives a convex objective's parameters toward the minimum.
TEST(Optimizer, AdamConverges) {
    OptimizerConfig cfg;
    cfg.enable_gradient_clipping = false;
    cfg.type = OptimizerType::Adam;
    AdamOptimizer<float> adam(cfg);

    EXPECT_LT(final_norm(adam, 0.05), 1e-2);
}

// 1.11: resize_state_ preserves the overlapping prefix and zeroes growth.
TEST(Optimizer, ResizeStatePreservesOverlap) {
    Tensor<float> s(std::vector<size_t>{4});
    for (size_t i = 0; i < 4; ++i) s.data()[i] = static_cast<float>(i + 1);

    dnn::training::resize_state_(s, 7u);
    ASSERT_EQ(s.size(), 7u);
    for (size_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(s.data()[i], static_cast<float>(i + 1));
    for (size_t i = 4; i < 7; ++i) EXPECT_FLOAT_EQ(s.data()[i], 0.0f);

    dnn::training::resize_state_(s, 2u);  // shrink truncates
    ASSERT_EQ(s.size(), 2u);
    EXPECT_FLOAT_EQ(s.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(s.data()[1], 2.0f);
}

// 1.11: Adam.resize keeps optimising after a parameter-count growth.
TEST(Optimizer, AdamResizeThenContinues) {
    OptimizerConfig cfg;
    cfg.enable_gradient_clipping = false;
    cfg.type = OptimizerType::Adam;
    AdamOptimizer<float> adam(cfg);
    adam.initialize(4, 1);
    adam.set_learning_rate(0.05f);

    Tensor<float> w(std::vector<size_t>{4}, 1.0f);
    Tensor<float> b(std::vector<size_t>{1}, 0.0f);
    for (int s = 0; s < 20; ++s) {
        Tensor<float> dw(std::vector<size_t>{4});
        for (size_t i = 0; i < 4; ++i) dw.data()[i] = 2.0f * w.data()[i];
        Tensor<float> db(std::vector<size_t>{1}, 0.0f);
        adam.update(w, b, dw, db);
    }

    adam.resize(8, 1);  // layer grew
    Tensor<float> w2(std::vector<size_t>{8}, 1.0f);
    for (size_t i = 0; i < 4; ++i) w2.data()[i] = w.data()[i];
    for (int s = 0; s < 200; ++s) {
        Tensor<float> dw(std::vector<size_t>{8});
        for (size_t i = 0; i < 8; ++i) dw.data()[i] = 2.0f * w2.data()[i];
        Tensor<float> db(std::vector<size_t>{1}, 0.0f);
        adam.update(w2, b, dw, db);
    }
    double n = 0.0;
    for (size_t i = 0; i < 8; ++i) n += w2.data()[i] * w2.data()[i];
    EXPECT_LT(std::sqrt(n), 1e-2);  // all 8 (old+new) converged
}

// Default trainer config must keep the legacy inline-SGD path
// (use_optimizer=false) so existing runs stay bit-for-bit.
TEST(Optimizer, TrainerDefaultDoesNotUseOptimizer) {
    TrainerConfig cfg;
    EXPECT_FALSE(cfg.use_optimizer);
}
