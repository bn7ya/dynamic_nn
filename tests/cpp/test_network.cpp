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

// 3.6: clone copies topology_version_ (and training_in_progress_).
TEST(Network, CloneCopiesTopologyVersion) {
    Network<float> net(make_config());
    auto copy = net.clone();
    EXPECT_EQ(copy->topology_version(), net.topology_version());
}

// 1.10: insert_layer rebuilds the following layer with a new fan-in
// but preserves the trained weights of the surviving columns.
TEST(Network, InsertLayerPreservesOverlappingWeights) {
    Network<float> net(make_config());
    ASSERT_EQ(net.num_layers(), 2u);

    // Stamp recognisable weights into the output layer (layers_[1]).
    auto& out = net.layer(1);
    const size_t old_out = out.output_size();
    const size_t old_in = out.input_size();
    for (size_t i = 0; i < old_out; ++i)
        for (size_t j = 0; j < old_in; ++j)
            out.weights().at(i, j) = static_cast<float>(100 + i * 10 + j);
    for (size_t i = 0; i < old_out; ++i)
        out.biases()[i] = static_cast<float>(7 + i);

    const size_t new_nodes = 2;  // narrower than old_in -> partial overlap
    net.insert_layer(1, new_nodes);

    auto& rebuilt = net.layer(2);  // old output layer, now after inserted
    ASSERT_EQ(rebuilt.input_size(), new_nodes);
    ASSERT_EQ(rebuilt.output_size(), old_out);
    const size_t keep = std::min(old_in, new_nodes);
    for (size_t i = 0; i < old_out; ++i) {
        for (size_t j = 0; j < keep; ++j)
            EXPECT_FLOAT_EQ(rebuilt.weights().at(i, j),
                            static_cast<float>(100 + i * 10 + j));
        EXPECT_FLOAT_EQ(rebuilt.biases()[i], static_cast<float>(7 + i));
    }
}
