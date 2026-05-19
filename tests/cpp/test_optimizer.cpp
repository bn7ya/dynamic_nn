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

// Default trainer config must keep the legacy inline-SGD path
// (use_optimizer=false) so existing runs stay bit-for-bit.
TEST(Optimizer, TrainerDefaultDoesNotUseOptimizer) {
    TrainerConfig cfg;
    EXPECT_FALSE(cfg.use_optimizer);
}
