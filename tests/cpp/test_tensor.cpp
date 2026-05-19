#include <gtest/gtest.h>

#include "dnn/core/tensor.hpp"

using dnn::core::Tensor;

TEST(Tensor, ConstructFillAndIndex) {
    Tensor<float> t(std::vector<size_t>{2, 3}, 1.5f);
    EXPECT_EQ(t.rank(), 2u);
    EXPECT_EQ(t.size(), 6u);
    EXPECT_FLOAT_EQ(t.at(0, 0), 1.5f);
    t.at(1, 2) = 4.0f;
    EXPECT_FLOAT_EQ(t.at(1, 2), 4.0f);
}

TEST(Tensor, TransposeRoundTrips) {
    Tensor<float> t(std::vector<size_t>{2, 3});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j) t.at(i, j) = static_cast<float>(i * 3 + j);

    Tensor<float> tt = t.transpose();
    ASSERT_EQ(tt.shape()[0], 3u);
    ASSERT_EQ(tt.shape()[1], 2u);
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 3; ++j)
            EXPECT_FLOAT_EQ(tt.at(j, i), t.at(i, j));
}
