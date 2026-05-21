#include <gtest/gtest.h>
#include <torch/torch.h>

#include "enn/modules/reversible_linear.hpp"


using enn::modules::ReversibleLinear;


TEST(ReversibleLinear, ForwardMatchesTorchLinearAllActive) {
    torch::manual_seed(0);
    ReversibleLinear rl(8, 4);
    auto x = torch::randn({3, 8});
    auto out = rl->forward(x);
    auto expected = torch::nn::functional::linear(x, rl->weight, rl->bias);
    EXPECT_TRUE(torch::allclose(out, expected));
}

TEST(ReversibleLinear, PruneThenAddRoundTrip) {
    torch::manual_seed(0);
    ReversibleLinear rl(8, 4);
    auto x = torch::randn({3, 8});
    auto full = rl->forward(x).clone();
    rl->prune_nodes({1, 2});
    auto pruned = rl->forward(x);
    EXPECT_TRUE(torch::allclose(pruned.index({torch::indexing::Slice(), 0}),
                                  full.index({torch::indexing::Slice(), 0})));
    EXPECT_TRUE(torch::allclose(pruned.index({torch::indexing::Slice(), 1}),
                                  torch::zeros({3})));
    auto reactivated = rl->add_nodes(2);
    EXPECT_EQ(reactivated, 2);
    auto after = rl->forward(x);
    EXPECT_TRUE(torch::allclose(after, full));
}

TEST(ReversibleLinear, TopologyVersionMonotonic) {
    ReversibleLinear rl(4, 4);
    auto v0 = rl->topology_version();
    rl->prune_nodes({0});
    auto v1 = rl->topology_version();
    rl->add_nodes(1);
    auto v2 = rl->topology_version();
    rl->compact();
    auto v3 = rl->topology_version();
    EXPECT_LT(v0, v1);
    EXPECT_LT(v1, v2);
    EXPECT_LT(v2, v3);
}

TEST(ReversibleLinear, CompactDropsInactiveRows) {
    ReversibleLinear rl(4, 6);
    rl->prune_nodes({1, 3, 5});
    auto dropped = rl->compact();
    EXPECT_EQ(dropped, 3);
    EXPECT_EQ(rl->out_features(), 3);
}
