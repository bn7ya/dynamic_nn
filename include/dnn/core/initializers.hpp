#pragma once

#include "tensor.hpp"
#include "random.hpp"
#include <cmath>

namespace dnn {
namespace core {

/**
 * Weight initialization strategies.
 */
enum class InitializerType {
    Zeros,
    Ones,
    Constant,
    Uniform,
    Normal,
    Xavier,          // aka Glorot
    XavierNormal,
    He,              // aka Kaiming
    HeNormal,
    LeCun,
    Orthogonal
};

/**
 * Base class for weight initializers.
 */
template<typename T = float>
class Initializer {
public:
    virtual ~Initializer() = default;

    /**
     * Initialize a tensor.
     */
    virtual void initialize(Tensor<T>& tensor, Random& rng) = 0;

    /**
     * Clone the initializer.
     */
    virtual std::unique_ptr<Initializer> clone() const = 0;

    /**
     * Factory method to create initializers.
     */
    static std::unique_ptr<Initializer> create(InitializerType type,
                                               size_t fan_in = 0,
                                               size_t fan_out = 0);
};

/**
 * Zero initialization.
 */
template<typename T = float>
class ZerosInitializer : public Initializer<T> {
public:
    void initialize(Tensor<T>& tensor, Random& /*rng*/) override {
        tensor.fill(T(0));
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<ZerosInitializer>();
    }
};

/**
 * Ones initialization.
 */
template<typename T = float>
class OnesInitializer : public Initializer<T> {
public:
    void initialize(Tensor<T>& tensor, Random& /*rng*/) override {
        tensor.fill(T(1));
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<OnesInitializer>();
    }
};

/**
 * Constant initialization.
 */
template<typename T = float>
class ConstantInitializer : public Initializer<T> {
public:
    explicit ConstantInitializer(T value) : value_(value) {}

    void initialize(Tensor<T>& tensor, Random& /*rng*/) override {
        tensor.fill(value_);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<ConstantInitializer>(value_);
    }

private:
    T value_;
};

/**
 * Uniform random initialization.
 */
template<typename T = float>
class UniformInitializer : public Initializer<T> {
public:
    UniformInitializer(T min_val = T(-1), T max_val = T(1))
        : min_val_(min_val), max_val_(max_val) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        rng.fill_uniform(tensor, min_val_, max_val_);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<UniformInitializer>(min_val_, max_val_);
    }

private:
    T min_val_;
    T max_val_;
};

/**
 * Normal random initialization.
 */
template<typename T = float>
class NormalInitializer : public Initializer<T> {
public:
    NormalInitializer(T mean = T(0), T stddev = T(1))
        : mean_(mean), stddev_(stddev) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        rng.fill_normal(tensor, mean_, stddev_);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<NormalInitializer>(mean_, stddev_);
    }

private:
    T mean_;
    T stddev_;
};

/**
 * Xavier/Glorot uniform initialization.
 * Good for tanh and sigmoid activations.
 * W ~ U[-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))]
 */
template<typename T = float>
class XavierInitializer : public Initializer<T> {
public:
    XavierInitializer(size_t fan_in, size_t fan_out)
        : fan_in_(fan_in), fan_out_(fan_out) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        T limit = std::sqrt(T(6) / (fan_in_ + fan_out_));
        rng.fill_uniform(tensor, -limit, limit);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<XavierInitializer>(fan_in_, fan_out_);
    }

private:
    size_t fan_in_;
    size_t fan_out_;
};

/**
 * Xavier/Glorot normal initialization.
 * W ~ N(0, sqrt(2/(fan_in + fan_out)))
 */
template<typename T = float>
class XavierNormalInitializer : public Initializer<T> {
public:
    XavierNormalInitializer(size_t fan_in, size_t fan_out)
        : fan_in_(fan_in), fan_out_(fan_out) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        T stddev = std::sqrt(T(2) / (fan_in_ + fan_out_));
        rng.fill_normal(tensor, T(0), stddev);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<XavierNormalInitializer>(fan_in_, fan_out_);
    }

private:
    size_t fan_in_;
    size_t fan_out_;
};

/**
 * He/Kaiming uniform initialization.
 * Good for ReLU activations.
 * W ~ U[-sqrt(6/fan_in), sqrt(6/fan_in)]
 */
template<typename T = float>
class HeInitializer : public Initializer<T> {
public:
    explicit HeInitializer(size_t fan_in) : fan_in_(fan_in) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        T limit = std::sqrt(T(6) / fan_in_);
        rng.fill_uniform(tensor, -limit, limit);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<HeInitializer>(fan_in_);
    }

private:
    size_t fan_in_;
};

/**
 * He/Kaiming normal initialization.
 * W ~ N(0, sqrt(2/fan_in))
 */
template<typename T = float>
class HeNormalInitializer : public Initializer<T> {
public:
    explicit HeNormalInitializer(size_t fan_in) : fan_in_(fan_in) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        T stddev = std::sqrt(T(2) / fan_in_);
        rng.fill_normal(tensor, T(0), stddev);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<HeNormalInitializer>(fan_in_);
    }

private:
    size_t fan_in_;
};

/**
 * LeCun initialization.
 * W ~ N(0, sqrt(1/fan_in))
 */
template<typename T = float>
class LeCunInitializer : public Initializer<T> {
public:
    explicit LeCunInitializer(size_t fan_in) : fan_in_(fan_in) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        T stddev = std::sqrt(T(1) / fan_in_);
        rng.fill_normal(tensor, T(0), stddev);
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<LeCunInitializer>(fan_in_);
    }

private:
    size_t fan_in_;
};

/**
 * Orthogonal initialization.
 * Useful for RNNs.
 */
template<typename T = float>
class OrthogonalInitializer : public Initializer<T> {
public:
    explicit OrthogonalInitializer(T gain = T(1)) : gain_(gain) {}

    void initialize(Tensor<T>& tensor, Random& rng) override {
        if (tensor.rank() != 2) {
            // For non-2D, just use normal initialization
            rng.fill_normal(tensor, T(0), gain_);
            return;
        }

        size_t rows = tensor.shape()[0];
        size_t cols = tensor.shape()[1];
        size_t n = std::max(rows, cols);

        // Generate random matrix
        Tensor<T> random(std::vector<size_t>{n, n});
        rng.fill_normal(random, T(0), T(1));

        // QR decomposition (simplified Gram-Schmidt)
        Tensor<T> Q(std::vector<size_t>{n, n});
        for (size_t i = 0; i < n; ++i) {
            // Copy column i
            for (size_t j = 0; j < n; ++j) {
                Q.at(j, i) = random.at(j, i);
            }

            // Subtract projections onto previous columns
            for (size_t k = 0; k < i; ++k) {
                T dot = T(0);
                for (size_t j = 0; j < n; ++j) {
                    dot += random.at(j, i) * Q.at(j, k);
                }
                for (size_t j = 0; j < n; ++j) {
                    Q.at(j, i) -= dot * Q.at(j, k);
                }
            }

            // Normalize
            T norm = T(0);
            for (size_t j = 0; j < n; ++j) {
                norm += Q.at(j, i) * Q.at(j, i);
            }
            norm = std::sqrt(norm);
            if (norm > T(1e-10)) {
                for (size_t j = 0; j < n; ++j) {
                    Q.at(j, i) /= norm;
                }
            }
        }

        // Copy relevant part to output
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                tensor.at(i, j) = Q.at(i, j) * gain_;
            }
        }
    }

    std::unique_ptr<Initializer<T>> clone() const override {
        return std::make_unique<OrthogonalInitializer>(gain_);
    }

private:
    T gain_;
};

// Factory implementation
template<typename T>
std::unique_ptr<Initializer<T>> Initializer<T>::create(InitializerType type,
                                                       size_t fan_in,
                                                       size_t fan_out) {
    switch (type) {
        case InitializerType::Zeros:
            return std::make_unique<ZerosInitializer<T>>();
        case InitializerType::Ones:
            return std::make_unique<OnesInitializer<T>>();
        case InitializerType::Constant:
            return std::make_unique<ConstantInitializer<T>>(T(0));
        case InitializerType::Uniform:
            return std::make_unique<UniformInitializer<T>>();
        case InitializerType::Normal:
            return std::make_unique<NormalInitializer<T>>();
        case InitializerType::Xavier:
            return std::make_unique<XavierInitializer<T>>(fan_in, fan_out);
        case InitializerType::XavierNormal:
            return std::make_unique<XavierNormalInitializer<T>>(fan_in, fan_out);
        case InitializerType::He:
            return std::make_unique<HeInitializer<T>>(fan_in);
        case InitializerType::HeNormal:
            return std::make_unique<HeNormalInitializer<T>>(fan_in);
        case InitializerType::LeCun:
            return std::make_unique<LeCunInitializer<T>>(fan_in);
        case InitializerType::Orthogonal:
            return std::make_unique<OrthogonalInitializer<T>>();
        default:
            return std::make_unique<XavierInitializer<T>>(fan_in, fan_out);
    }
}

/**
 * Helper function to automatically choose initializer based on activation.
 */
template<typename T = float>
std::unique_ptr<Initializer<T>> auto_initializer(const std::string& activation,
                                                  size_t fan_in, size_t fan_out) {
    if (activation == "relu" || activation == "leaky_relu") {
        return std::make_unique<HeNormalInitializer<T>>(fan_in);
    } else if (activation == "tanh" || activation == "sigmoid") {
        return std::make_unique<XavierNormalInitializer<T>>(fan_in, fan_out);
    } else if (activation == "selu") {
        return std::make_unique<LeCunInitializer<T>>(fan_in);
    } else {
        // Default: Xavier
        return std::make_unique<XavierNormalInitializer<T>>(fan_in, fan_out);
    }
}

} // namespace core
} // namespace dnn
