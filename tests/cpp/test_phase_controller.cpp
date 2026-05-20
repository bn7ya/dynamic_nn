#include <gtest/gtest.h>
#include <torch/torch.h>

#include "enn/modules/reversible_network.hpp"
#include "enn/training/phase_controller.hpp"


using enn::modules::HiddenActivation;
using enn::modules::ReversibleNetwork;
using enn::modules::ReversibleNetworkConfig;
using enn::training::CostFunction;
using enn::training::PhaseController;


TEST(PhaseController, EndToEndFit) {
    torch::manual_seed(0);
    ReversibleNetworkConfig cfg;
    cfg.input_features = 16;
    cfg.output_features = 4;
    cfg.hidden_sizes = {8};
    ReversibleNetwork net(cfg);
    auto X = torch::randn({64, 16});
    auto y = torch::eye(4).index({torch::randint(0, 4, {64},
                                                   torch::kInt64)});
    PhaseController pc(*net, CostFunction::CrossEntropy);
    auto result = pc.fit(X, y);
    EXPECT_GT(result.epochs_completed, 0);
    EXPECT_FALSE(result.cost_trajectory.empty());
    EXPECT_GT(result.parameter_count_final, 0);
}
