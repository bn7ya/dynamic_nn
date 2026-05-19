#pragma once

#include "../core/tensor.hpp"
#include <memory>
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

namespace dnn {
namespace training {

using core::Tensor;

/**
 * Optimizer type enumeration.
 */
enum class OptimizerType {
    SGD,          // Basic stochastic gradient descent
    SGDMomentum,  // SGD with momentum
    Adam,         // Adaptive moment estimation
    RMSprop       // Root mean square propagation
};

/**
 * Optimizer configuration.
 * All parameters have sensible defaults matching Python implementation.
 */
struct OptimizerConfig {
    OptimizerType type = OptimizerType::Adam;

    // Learning rate
    double learning_rate = 0.01;

    // SGD Momentum parameters
    double momentum = 0.9;

    // Adam parameters
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    // RMSprop parameters
    double decay_rate = 0.99;

    // Gradient clipping (matches Python: gradient_clip_value = 1.0)
    double gradient_clip = 1.0;
    bool enable_gradient_clipping = true;

    // Weight decay (L2 regularization)
    double weight_decay = 0.0;
};

/**
 * Abstract optimizer interface.
 * All optimizers follow the same update pattern.
 */
template<typename T>
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /**
     * Update weights and biases using computed gradients.
     * @param weights Weight tensor to update (modified in-place)
     * @param biases Bias tensor to update (modified in-place)
     * @param weight_grads Gradient of loss w.r.t weights
     * @param bias_grads Gradient of loss w.r.t biases
     */
    virtual void update(Tensor<T>& weights, Tensor<T>& biases,
                       const Tensor<T>& weight_grads, const Tensor<T>& bias_grads) = 0;

    /**
     * Increment the timestep (for Adam).
     */
    virtual void step() = 0;

    /**
     * Reset optimizer state (momentum, etc.).
     */
    virtual void reset() = 0;

    /**
     * Get/set learning rate.
     */
    virtual T get_learning_rate() const = 0;
    virtual void set_learning_rate(T lr) = 0;

    /**
     * Initialize state for a specific layer size.
     * Must be called before first update.
     */
    virtual void initialize(size_t weight_size, size_t bias_size) = 0;

    /**
     * Resize state after the layer's parameter count changes (e.g.
     * Layer::add_nodes grew the weight matrix). Existing state for the
     * overlapping prefix is preserved; newly-added entries start at
     * zero (no momentum/variance history — correct, they are new).
     * Default: no-op (stateless optimizers).
     */
    virtual void resize(size_t weight_size, size_t bias_size) {
        (void)weight_size; (void)bias_size;
    }

    /**
     * Factory method to create optimizer from config.
     */
    static std::unique_ptr<Optimizer<T>> create(const OptimizerConfig& config);
};

/**
 * Resize a 1-D state tensor, preserving the first min(old,new) elements
 * and zero-filling any growth. Shrinking simply truncates.
 */
template<typename T>
inline void resize_state_(Tensor<T>& state, size_t new_size) {
    if (state.size() == new_size) return;
    Tensor<T> resized(std::vector<size_t>{new_size}, T(0));
    const size_t keep = std::min(state.size(), new_size);
    for (size_t i = 0; i < keep; ++i) resized.data()[i] = state.data()[i];
    state = std::move(resized);
}

/**
 * Gradient clipping utility.
 * Matches Python: dW = np.clip(dW, -clip_value, clip_value)
 */
template<typename T>
void clip_gradients(Tensor<T>& grads, T clip_value) {
    T* data = grads.data();
    const size_t n = grads.size();
    for (size_t i = 0; i < n; ++i) {
        data[i] = std::max(static_cast<T>(-clip_value),
                          std::min(static_cast<T>(clip_value), data[i]));
    }
}

/**
 * SGD Optimizer - Basic Stochastic Gradient Descent.
 * Matches Python: W -= lr * dW
 */
template<typename T>
class SGDOptimizer : public Optimizer<T> {
public:
    explicit SGDOptimizer(const OptimizerConfig& config)
        : learning_rate_(static_cast<T>(config.learning_rate))
        , gradient_clip_(static_cast<T>(config.gradient_clip))
        , enable_clipping_(config.enable_gradient_clipping)
        , weight_decay_(static_cast<T>(config.weight_decay)) {}

    void update(Tensor<T>& weights, Tensor<T>& biases,
               const Tensor<T>& weight_grads, const Tensor<T>& bias_grads) override {
        // Create mutable copies for clipping
        Tensor<T> dW = weight_grads.clone();
        Tensor<T> db = bias_grads.clone();

        // Apply gradient clipping if enabled
        if (enable_clipping_) {
            clip_gradients(dW, gradient_clip_);
            clip_gradients(db, gradient_clip_);
        }

        T* w_data = weights.data();
        T* b_data = biases.data();
        const T* dw_data = dW.data();
        const T* db_data = db.data();

        // Update weights: W -= lr * dW
        for (size_t i = 0; i < weights.size(); ++i) {
            T grad = dw_data[i];
            if (weight_decay_ > 0) {
                grad += weight_decay_ * w_data[i];
            }
            w_data[i] -= learning_rate_ * grad;
        }

        // Update biases: b -= lr * db
        for (size_t i = 0; i < biases.size(); ++i) {
            b_data[i] -= learning_rate_ * db_data[i];
        }
    }

    void step() override {} // No-op for basic SGD
    void reset() override {} // No state to reset
    void initialize(size_t, size_t) override {} // No state to initialize

    T get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(T lr) override { learning_rate_ = lr; }

private:
    T learning_rate_;
    T gradient_clip_;
    bool enable_clipping_;
    T weight_decay_;
};

/**
 * SGD with Momentum Optimizer.
 * v = momentum * v + lr * dW
 * W -= v
 */
template<typename T>
class SGDMomentumOptimizer : public Optimizer<T> {
public:
    explicit SGDMomentumOptimizer(const OptimizerConfig& config)
        : learning_rate_(static_cast<T>(config.learning_rate))
        , momentum_(static_cast<T>(config.momentum))
        , gradient_clip_(static_cast<T>(config.gradient_clip))
        , enable_clipping_(config.enable_gradient_clipping)
        , weight_decay_(static_cast<T>(config.weight_decay))
        , initialized_(false) {}

    void initialize(size_t weight_size, size_t bias_size) override {
        velocity_w_ = Tensor<T>(std::vector<size_t>{weight_size}, T(0));
        velocity_b_ = Tensor<T>(std::vector<size_t>{bias_size}, T(0));
        initialized_ = true;
    }

    void resize(size_t weight_size, size_t bias_size) override {
        if (!initialized_) { initialize(weight_size, bias_size); return; }
        resize_state_(velocity_w_, weight_size);
        resize_state_(velocity_b_, bias_size);
    }

    void update(Tensor<T>& weights, Tensor<T>& biases,
               const Tensor<T>& weight_grads, const Tensor<T>& bias_grads) override {
        if (!initialized_) {
            initialize(weights.size(), biases.size());
        }

        // Create mutable copies for clipping
        Tensor<T> dW = weight_grads.clone();
        Tensor<T> db = bias_grads.clone();

        if (enable_clipping_) {
            clip_gradients(dW, gradient_clip_);
            clip_gradients(db, gradient_clip_);
        }

        T* w_data = weights.data();
        T* b_data = biases.data();
        const T* dw_data = dW.data();
        const T* db_data = db.data();
        T* vw_data = velocity_w_.data();
        T* vb_data = velocity_b_.data();

        // Update weight velocity and weights
        for (size_t i = 0; i < weights.size(); ++i) {
            T grad = dw_data[i];
            if (weight_decay_ > 0) {
                grad += weight_decay_ * w_data[i];
            }
            vw_data[i] = momentum_ * vw_data[i] + learning_rate_ * grad;
            w_data[i] -= vw_data[i];
        }

        // Update bias velocity and biases
        for (size_t i = 0; i < biases.size(); ++i) {
            vb_data[i] = momentum_ * vb_data[i] + learning_rate_ * db_data[i];
            b_data[i] -= vb_data[i];
        }
    }

    void step() override {}

    void reset() override {
        if (initialized_) {
            velocity_w_.fill(T(0));
            velocity_b_.fill(T(0));
        }
    }

    T get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(T lr) override { learning_rate_ = lr; }

private:
    T learning_rate_;
    T momentum_;
    T gradient_clip_;
    bool enable_clipping_;
    T weight_decay_;
    bool initialized_;
    Tensor<T> velocity_w_;
    Tensor<T> velocity_b_;
};

/**
 * Adam Optimizer - Adaptive Moment Estimation.
 *
 * Algorithm:
 *   m = beta1 * m + (1 - beta1) * grad
 *   v = beta2 * v + (1 - beta2) * grad^2
 *   m_hat = m / (1 - beta1^t)
 *   v_hat = v / (1 - beta2^t)
 *   W -= lr * m_hat / (sqrt(v_hat) + epsilon)
 */
template<typename T>
class AdamOptimizer : public Optimizer<T> {
public:
    explicit AdamOptimizer(const OptimizerConfig& config)
        : learning_rate_(static_cast<T>(config.learning_rate))
        , beta1_(static_cast<T>(config.beta1))
        , beta2_(static_cast<T>(config.beta2))
        , epsilon_(static_cast<T>(config.epsilon))
        , gradient_clip_(static_cast<T>(config.gradient_clip))
        , enable_clipping_(config.enable_gradient_clipping)
        , weight_decay_(static_cast<T>(config.weight_decay))
        , timestep_(0)
        , initialized_(false) {}

    void initialize(size_t weight_size, size_t bias_size) override {
        // First moment (mean)
        m_w_ = Tensor<T>(std::vector<size_t>{weight_size}, T(0));
        m_b_ = Tensor<T>(std::vector<size_t>{bias_size}, T(0));

        // Second moment (variance)
        v_w_ = Tensor<T>(std::vector<size_t>{weight_size}, T(0));
        v_b_ = Tensor<T>(std::vector<size_t>{bias_size}, T(0));

        initialized_ = true;
    }

    void resize(size_t weight_size, size_t bias_size) override {
        if (!initialized_) { initialize(weight_size, bias_size); return; }
        // Preserve moments for surviving params; new params get no
        // history. timestep_ (bias correction) is intentionally kept.
        resize_state_(m_w_, weight_size);
        resize_state_(m_b_, bias_size);
        resize_state_(v_w_, weight_size);
        resize_state_(v_b_, bias_size);
    }

    void update(Tensor<T>& weights, Tensor<T>& biases,
               const Tensor<T>& weight_grads, const Tensor<T>& bias_grads) override {
        if (!initialized_) {
            initialize(weights.size(), biases.size());
        }

        // Increment timestep (must be done before update for correct bias correction)
        ++timestep_;

        // Create mutable copies for clipping
        Tensor<T> dW = weight_grads.clone();
        Tensor<T> db = bias_grads.clone();

        if (enable_clipping_) {
            clip_gradients(dW, gradient_clip_);
            clip_gradients(db, gradient_clip_);
        }

        // Bias correction factors
        T beta1_t = std::pow(beta1_, static_cast<T>(timestep_));
        T beta2_t = std::pow(beta2_, static_cast<T>(timestep_));
        T bias_correction1 = T(1) - beta1_t;
        T bias_correction2 = T(1) - beta2_t;

        // Update weights
        T* w_data = weights.data();
        const T* dw_data = dW.data();
        T* mw_data = m_w_.data();
        T* vw_data = v_w_.data();

        for (size_t i = 0; i < weights.size(); ++i) {
            T grad = dw_data[i];
            if (weight_decay_ > 0) {
                grad += weight_decay_ * w_data[i];
            }

            // Update biased first moment estimate
            mw_data[i] = beta1_ * mw_data[i] + (T(1) - beta1_) * grad;

            // Update biased second raw moment estimate
            vw_data[i] = beta2_ * vw_data[i] + (T(1) - beta2_) * grad * grad;

            // Compute bias-corrected first moment estimate
            T m_hat = mw_data[i] / bias_correction1;

            // Compute bias-corrected second raw moment estimate
            T v_hat = vw_data[i] / bias_correction2;

            // Update parameters
            w_data[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }

        // Update biases
        T* b_data = biases.data();
        const T* db_data = db.data();
        T* mb_data = m_b_.data();
        T* vb_data = v_b_.data();

        for (size_t i = 0; i < biases.size(); ++i) {
            T grad = db_data[i];

            mb_data[i] = beta1_ * mb_data[i] + (T(1) - beta1_) * grad;
            vb_data[i] = beta2_ * vb_data[i] + (T(1) - beta2_) * grad * grad;

            T m_hat = mb_data[i] / bias_correction1;
            T v_hat = vb_data[i] / bias_correction2;

            b_data[i] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
        }
    }

    void step() override {
        // Timestep is incremented in update()
    }

    void reset() override {
        timestep_ = 0;
        if (initialized_) {
            m_w_.fill(T(0));
            m_b_.fill(T(0));
            v_w_.fill(T(0));
            v_b_.fill(T(0));
        }
    }

    T get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(T lr) override { learning_rate_ = lr; }

private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    T gradient_clip_;
    bool enable_clipping_;
    T weight_decay_;
    size_t timestep_;
    bool initialized_;

    // First moment estimates
    Tensor<T> m_w_;
    Tensor<T> m_b_;

    // Second moment estimates
    Tensor<T> v_w_;
    Tensor<T> v_b_;
};

/**
 * RMSprop Optimizer - Root Mean Square Propagation.
 *
 * Algorithm:
 *   cache = decay_rate * cache + (1 - decay_rate) * grad^2
 *   W -= lr * grad / (sqrt(cache) + epsilon)
 */
template<typename T>
class RMSpropOptimizer : public Optimizer<T> {
public:
    explicit RMSpropOptimizer(const OptimizerConfig& config)
        : learning_rate_(static_cast<T>(config.learning_rate))
        , decay_rate_(static_cast<T>(config.decay_rate))
        , epsilon_(static_cast<T>(config.epsilon))
        , gradient_clip_(static_cast<T>(config.gradient_clip))
        , enable_clipping_(config.enable_gradient_clipping)
        , weight_decay_(static_cast<T>(config.weight_decay))
        , initialized_(false) {}

    void initialize(size_t weight_size, size_t bias_size) override {
        cache_w_ = Tensor<T>(std::vector<size_t>{weight_size}, T(0));
        cache_b_ = Tensor<T>(std::vector<size_t>{bias_size}, T(0));
        initialized_ = true;
    }

    void resize(size_t weight_size, size_t bias_size) override {
        if (!initialized_) { initialize(weight_size, bias_size); return; }
        resize_state_(cache_w_, weight_size);
        resize_state_(cache_b_, bias_size);
    }

    void update(Tensor<T>& weights, Tensor<T>& biases,
               const Tensor<T>& weight_grads, const Tensor<T>& bias_grads) override {
        if (!initialized_) {
            initialize(weights.size(), biases.size());
        }

        // Create mutable copies for clipping
        Tensor<T> dW = weight_grads.clone();
        Tensor<T> db = bias_grads.clone();

        if (enable_clipping_) {
            clip_gradients(dW, gradient_clip_);
            clip_gradients(db, gradient_clip_);
        }

        // Update weights
        T* w_data = weights.data();
        const T* dw_data = dW.data();
        T* cache_w_data = cache_w_.data();

        for (size_t i = 0; i < weights.size(); ++i) {
            T grad = dw_data[i];
            if (weight_decay_ > 0) {
                grad += weight_decay_ * w_data[i];
            }

            // Update cache
            cache_w_data[i] = decay_rate_ * cache_w_data[i] +
                             (T(1) - decay_rate_) * grad * grad;

            // Update parameters
            w_data[i] -= learning_rate_ * grad / (std::sqrt(cache_w_data[i]) + epsilon_);
        }

        // Update biases
        T* b_data = biases.data();
        const T* db_data = db.data();
        T* cache_b_data = cache_b_.data();

        for (size_t i = 0; i < biases.size(); ++i) {
            T grad = db_data[i];

            cache_b_data[i] = decay_rate_ * cache_b_data[i] +
                             (T(1) - decay_rate_) * grad * grad;

            b_data[i] -= learning_rate_ * grad / (std::sqrt(cache_b_data[i]) + epsilon_);
        }
    }

    void step() override {}

    void reset() override {
        if (initialized_) {
            cache_w_.fill(T(0));
            cache_b_.fill(T(0));
        }
    }

    T get_learning_rate() const override { return learning_rate_; }
    void set_learning_rate(T lr) override { learning_rate_ = lr; }

private:
    T learning_rate_;
    T decay_rate_;
    T epsilon_;
    T gradient_clip_;
    bool enable_clipping_;
    T weight_decay_;
    bool initialized_;

    Tensor<T> cache_w_;
    Tensor<T> cache_b_;
};

/**
 * Factory method implementation.
 */
template<typename T>
std::unique_ptr<Optimizer<T>> Optimizer<T>::create(const OptimizerConfig& config) {
    switch (config.type) {
        case OptimizerType::SGD:
            return std::make_unique<SGDOptimizer<T>>(config);
        case OptimizerType::SGDMomentum:
            return std::make_unique<SGDMomentumOptimizer<T>>(config);
        case OptimizerType::Adam:
            return std::make_unique<AdamOptimizer<T>>(config);
        case OptimizerType::RMSprop:
            return std::make_unique<RMSpropOptimizer<T>>(config);
        default:
            throw std::invalid_argument("Unknown optimizer type");
    }
}

// Explicit template instantiations declaration
extern template class Optimizer<float>;
extern template class Optimizer<double>;
extern template class SGDOptimizer<float>;
extern template class SGDOptimizer<double>;
extern template class SGDMomentumOptimizer<float>;
extern template class SGDMomentumOptimizer<double>;
extern template class AdamOptimizer<float>;
extern template class AdamOptimizer<double>;
extern template class RMSpropOptimizer<float>;
extern template class RMSpropOptimizer<double>;

} // namespace training
} // namespace dnn
