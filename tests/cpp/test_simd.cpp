#include <gtest/gtest.h>

#include <vector>

#include "dnn/simd/simd_ops.hpp"

using dnn::simd::SIMDOps;

TEST(SIMD, AddMatchesScalarReference) {
    auto ops = SIMDOps<float>::create();
    ASSERT_NE(ops, nullptr);

    const size_t n = 37;  // deliberately not a SIMD-width multiple
    std::vector<float> a(n), b(n), c(n);
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i + 1);
    }

    ops->add(a.data(), b.data(), c.data(), n);

    for (size_t i = 0; i < n; ++i)
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
}

TEST(SIMD, FillSetsEveryElement) {
    auto ops = SIMDOps<float>::create();
    ASSERT_NE(ops, nullptr);

    std::vector<float> v(20, 0.0f);
    ops->fill(v.data(), 3.5f, v.size());
    for (float x : v) EXPECT_FLOAT_EQ(x, 3.5f);
}
