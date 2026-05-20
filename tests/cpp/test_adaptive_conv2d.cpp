#include <gtest/gtest.h>
#include <torch/torch.h>

#include "enn/modules/adaptive_conv2d.hpp"


using enn::modules::AdaptiveConv2d;


TEST(AdaptiveConv2d, ForwardShape) {
    AdaptiveConv2d c(3, 8, {3, 5, 7});
    auto x = torch::randn({2, 3, 16, 16});
    auto y = c->forward(x);
    EXPECT_EQ(y.size(0), 2);
    EXPECT_EQ(y.size(1), 8);
}

TEST(AdaptiveConv2d, PruneKeepsAtLeastOne) {
    AdaptiveConv2d c(3, 4, {3, 5, 7});
    c->prune_smallest_candidate();
    c->prune_smallest_candidate();
    c->prune_smallest_candidate();
    EXPECT_GE(c->candidate_mask().sum().item<double>(), 1.0);
}

TEST(AdaptiveConv2d, TopologyVersionIncrementsOnPrune) {
    AdaptiveConv2d c(3, 4, {3, 5});
    auto v0 = c->topology_version();
    c->prune_smallest_candidate();
    EXPECT_GT(c->topology_version(), v0);
}
