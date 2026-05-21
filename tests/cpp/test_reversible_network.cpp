#include <gtest/gtest.h>
#include <torch/torch.h>

#include "enn/modules/reversible_network.hpp"


using enn::modules::HiddenActivation;
using enn::modules::ReversibleNetwork;
using enn::modules::ReversibleNetworkConfig;


TEST(ReversibleNetwork, ForwardShape) {
    ReversibleNetworkConfig cfg;
    cfg.input_features = 8;
    cfg.output_features = 3;
    cfg.hidden_sizes = {6, 4};
    cfg.hidden_activation = HiddenActivation::ReLU;
    ReversibleNetwork net(cfg);
    auto x = torch::randn({5, 8});
    auto y = net->forward(x);
    EXPECT_EQ(y.size(0), 5);
    EXPECT_EQ(y.size(1), 3);
}

TEST(ReversibleNetwork, CompactReducesTopology) {
    ReversibleNetworkConfig cfg;
    cfg.input_features = 6;
    cfg.output_features = 2;
    cfg.hidden_sizes = {4, 4};
    ReversibleNetwork net(cfg);
    net->layer(0).prune_nodes({0, 1});
    auto dropped = net->compact();
    EXPECT_GE(dropped, 2);
    auto x = torch::randn({2, 6});
    auto y = net->forward(x);
    EXPECT_EQ(y.size(0), 2);
    EXPECT_EQ(y.size(1), 2);
}

TEST(ReversibleNetwork, TopologyVersionMonotonic) {
    ReversibleNetworkConfig cfg;
    cfg.input_features = 4;
    cfg.output_features = 2;
    cfg.hidden_sizes = {3};
    ReversibleNetwork net(cfg);
    auto v0 = net->topology_version();
    net->insert_layer(1, 3);
    auto v1 = net->topology_version();
    EXPECT_GT(v1, v0);
}
