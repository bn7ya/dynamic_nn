#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

#include "dnn/memory/pool_allocator.hpp"

using dnn::memory::PoolAllocator;

namespace {
// A spread of sizes that lands in distinct size classes (64B .. 1MB).
const std::vector<size_t> kSizes = {
    32, 100, 200, 500, 1000, 4000, 16000, 64000, 250000, 1000000};
}  // namespace

// After a full alloc+free round, the pool holds the freed blocks in the
// correct size classes. A second identical round must be satisfied
// entirely from the pool -- total bytes pulled from the system must not
// grow. The old "first non-empty pool" bug returned blocks to the wrong
// class, so the matching class stayed empty and the second round forced
// fresh system allocations, growing bytes_allocated().
TEST(PoolAllocator, RoundTripReuseDoesNotGrowPool) {
    auto& pool = PoolAllocator::instance();

    auto round = [&]() {
        std::vector<void*> ptrs;
        for (size_t s : kSizes) ptrs.push_back(pool.allocate(s));
        for (void* p : ptrs) ASSERT_NE(p, nullptr);
        for (void* p : ptrs) pool.deallocate(p);
    };

    round();  // warm: populate every size class's free list
    size_t baseline = pool.bytes_allocated();
    for (int i = 0; i < 50; ++i) round();
    EXPECT_EQ(pool.bytes_allocated(), baseline)
        << "pool grew -> blocks returned to wrong size class";
}

TEST(PoolAllocator, ConcurrentMixedAllocFreeIsConsistent) {
    auto& pool = PoolAllocator::instance();
    auto worker = [&]() {
        for (int it = 0; it < 2000; ++it) {
            std::vector<void*> ptrs;
            for (size_t s : kSizes) {
                void* p = pool.allocate(s);
                ASSERT_NE(p, nullptr);
                // Touch the memory: a wrong-class block would be smaller
                // than requested and this would corrupt the heap.
                std::memset(p, 0xAB, s);
                ptrs.push_back(p);
            }
            for (void* p : ptrs) pool.deallocate(p);
        }
    };
    std::vector<std::thread> ts;
    for (int i = 0; i < 4; ++i) ts.emplace_back(worker);
    for (auto& t : ts) t.join();
    SUCCEED();
}
