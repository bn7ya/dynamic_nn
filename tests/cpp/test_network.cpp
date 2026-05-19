#include <gtest/gtest.h>

#include "dnn/core/network.hpp"

using dnn::core::Network;
using dnn::core::NetworkConfig;
using dnn::core::Tensor;

namespace {
NetworkConfig make_config() {
    NetworkConfig cfg;
    cfg.seed = 42;
    cfg.input_shape = {8};
    cfg.output_size = 3;
    return cfg;
}
}  // namespace

TEST(Network, ForwardProducesOutputShape) {
    Network<float> net(make_config());
    Tensor<float> x(std::vector<size_t>{8}, 0.1f);
    auto y = net.forward(x);
    EXPECT_EQ(y.size(), 3u);
}

TEST(Network, CloneIsIndependent) {
    Network<float> net(make_config());
    auto copy = net.clone();
    ASSERT_NE(copy, nullptr);
    EXPECT_EQ(copy->num_layers(), net.num_layers());
}
