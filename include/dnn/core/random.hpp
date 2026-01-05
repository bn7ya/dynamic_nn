#pragma once

#include "tensor.hpp"
#include <random>
#include <cstdint>

namespace dnn {
namespace core {

/**
 * Thread-safe random number generator with seeding support.
 * This is the primary source of randomness for the network.
 */
class Random {
public:
    /**
     * Construct with seed.
     * @param seed The random seed - the ONLY hyperparameter required.
     */
    explicit Random(uint64_t seed)
        : generator_(seed)
        , seed_(seed) {}

    /**
     * Get the original seed.
     */
    uint64_t seed() const { return seed_; }

    /**
     * Reset to original seed.
     */
    void reset() {
        generator_.seed(seed_);
    }

    /**
     * Reseed with new value.
     */
    void reseed(uint64_t new_seed) {
        seed_ = new_seed;
        generator_.seed(new_seed);
    }

    /**
     * Generate uniform random float in [0, 1).
     */
    template<typename T = float>
    T uniform() {
        std::uniform_real_distribution<T> dist(T(0), T(1));
        return dist(generator_);
    }

    /**
     * Generate uniform random float in [min, max).
     */
    template<typename T = float>
    T uniform(T min_val, T max_val) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        return dist(generator_);
    }

    /**
     * Generate normal (Gaussian) random float.
     */
    template<typename T = float>
    T normal(T mean = T(0), T stddev = T(1)) {
        std::normal_distribution<T> dist(mean, stddev);
        return dist(generator_);
    }

    /**
     * Generate random integer in [min, max].
     */
    int randint(int min_val, int max_val) {
        std::uniform_int_distribution<int> dist(min_val, max_val);
        return dist(generator_);
    }

    /**
     * Generate random size_t in [0, max).
     */
    size_t randsize(size_t max_val) {
        if (max_val == 0) return 0;
        std::uniform_int_distribution<size_t> dist(0, max_val - 1);
        return dist(generator_);
    }

    /**
     * Fill tensor with uniform random values.
     */
    template<typename T>
    void fill_uniform(Tensor<T>& tensor, T min_val = T(0), T max_val = T(1)) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        T* data = tensor.data();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = dist(generator_);
        }
    }

    /**
     * Fill tensor with normal random values.
     */
    template<typename T>
    void fill_normal(Tensor<T>& tensor, T mean = T(0), T stddev = T(1)) {
        std::normal_distribution<T> dist(mean, stddev);
        T* data = tensor.data();
        for (size_t i = 0; i < tensor.size(); ++i) {
            data[i] = dist(generator_);
        }
    }

    /**
     * Shuffle indices in-place.
     */
    void shuffle(std::vector<size_t>& indices) {
        for (size_t i = indices.size() - 1; i > 0; --i) {
            size_t j = randsize(i + 1);
            std::swap(indices[i], indices[j]);
        }
    }

    /**
     * Generate random permutation of [0, n).
     */
    std::vector<size_t> permutation(size_t n) {
        std::vector<size_t> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = i;
        }
        shuffle(result);
        return result;
    }

    /**
     * Random choice from indices.
     */
    size_t choice(size_t n) {
        return randsize(n);
    }

    /**
     * Random choice of k elements from [0, n) without replacement.
     */
    std::vector<size_t> choice(size_t n, size_t k) {
        if (k > n) k = n;

        std::vector<size_t> result;
        result.reserve(k);

        if (k > n / 2) {
            // For large k, shuffle and take first k
            auto perm = permutation(n);
            result.assign(perm.begin(), perm.begin() + k);
        } else {
            // For small k, rejection sampling
            std::unordered_set<size_t> selected;
            while (result.size() < k) {
                size_t idx = randsize(n);
                if (selected.find(idx) == selected.end()) {
                    selected.insert(idx);
                    result.push_back(idx);
                }
            }
        }
        return result;
    }

    /**
     * Bernoulli trial.
     */
    bool bernoulli(double p = 0.5) {
        return uniform<double>() < p;
    }

    /**
     * Get underlying generator (for advanced use).
     */
    std::mt19937_64& generator() { return generator_; }

private:
    std::mt19937_64 generator_;
    uint64_t seed_;
};

/**
 * Global random instance for convenience.
 * Should be seeded once at program start.
 */
class GlobalRandom {
public:
    static Random& instance(uint64_t seed = 42) {
        static Random rng(seed);
        return rng;
    }

    static void seed(uint64_t seed) {
        instance().reseed(seed);
    }
};

} // namespace core
} // namespace dnn
