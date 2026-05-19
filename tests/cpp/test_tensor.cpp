#include <gtest/gtest.h>

#include <type_traits>

#include "dnn/core/tensor.hpp"

using dnn::core::Tensor;

// 1.6: view() must not be invocable on a const Tensor (no const_cast
// escape hatch). const_view() is the read-only path.
static_assert(
    !std::is_invocable_v<decltype(&Tensor<float>::view), const Tensor<float>&>,
    "Tensor::view() must be non-const");

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

// 2.8: exercise the tiled path with a large non-square matrix that
// spans many 32x32 tiles, including partial edge tiles.
TEST(Tensor, TiledTransposeLargeNonSquare) {
    const size_t R = 100, C = 70;
    Tensor<float> t(std::vector<size_t>{R, C});
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            t.at(i, j) = static_cast<float>(i * C + j);

    Tensor<float> tt = t.transpose();
    ASSERT_EQ(tt.shape()[0], C);
    ASSERT_EQ(tt.shape()[1], R);
    for (size_t i = 0; i < R; ++i)
        for (size_t j = 0; j < C; ++j)
            EXPECT_FLOAT_EQ(tt.at(j, i), t.at(i, j));
}

// 1.6: const_view aliases storage read-only; view() mutates in place.
TEST(Tensor, ViewMutatesConstViewReadsOnly) {
    Tensor<float> t(std::vector<size_t>{2, 2}, 1.0f);

    Tensor<float> v = t.view();
    v.at(0, 0) = 9.0f;
    EXPECT_FLOAT_EQ(t.at(0, 0), 9.0f);  // shares storage

    const Tensor<float>& ct = t;
    auto cv = ct.const_view();
    EXPECT_FLOAT_EQ(cv.at(0, 0), 9.0f);
    EXPECT_EQ(cv.size(), t.size());
}
