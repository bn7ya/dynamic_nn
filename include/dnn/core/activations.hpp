#pragma once

#include "tensor.hpp"
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>

namespace dnn {
namespace core {

/**
 * Activation function types.
 */
enum class ActivationType {
    Linear,
    ReLU,
    LeakyReLU,
    ELU,
    SELU,
    Sigmoid,
    Tanh,
    Softmax,
    Swish,
    GELU,
    Softplus
};

/**
 * Base class for activation functions.
 */
template<typename T = float>
class Activation {
public:
    virtual ~Activation() = default;

    /**
     * Apply activation function.
     */
    virtual Tensor<T> forward(const Tensor<T>& input) const = 0;

    /**
     * Compute gradient.
     * @param input Original input to forward pass
     * @param output Output from forward pass
     * @param grad_output Gradient from next layer
     * @return Gradient with respect to input
     */
    virtual Tensor<T> backward(const Tensor<T>& input,
                               const Tensor<T>& output,
                               const Tensor<T>& grad_output) const = 0;

    /**
     * Get activation type.
     */
    virtual ActivationType type() const = 0;

    /**
     * Get activation name.
     */
    virtual std::string name() const = 0;

    /**
     * Clone the activation.
     */
    virtual std::unique_ptr<Activation> clone() const = 0;

    /**
     * Factory method.
     */
    static std::unique_ptr<Activation> create(ActivationType type);
    static std::unique_ptr<Activation> create(const std::string& name);
};

/**
 * Linear (identity) activation.
 */
template<typename T = float>
class LinearActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return input.clone();
    }

    Tensor<T> backward(const Tensor<T>& /*input*/,
                       const Tensor<T>& /*output*/,
                       const Tensor<T>& grad_output) const override {
        return grad_output.clone();
    }

    ActivationType type() const override { return ActivationType::Linear; }
    std::string name() const override { return "linear"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<LinearActivation>();
    }
};

/**
 * ReLU activation: f(x) = max(0, x)
 */
template<typename T = float>
class ReLUActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return input.apply([](T x) { return x > T(0) ? x : T(0); });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& /*output*/,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = input[i] > T(0) ? grad_output[i] : T(0);
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::ReLU; }
    std::string name() const override { return "relu"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<ReLUActivation>();
    }
};

/**
 * Leaky ReLU activation: f(x) = x if x > 0, else alpha * x
 */
template<typename T = float>
class LeakyReLUActivation : public Activation<T> {
public:
    explicit LeakyReLUActivation(T alpha = T(0.01)) : alpha_(alpha) {}

    Tensor<T> forward(const Tensor<T>& input) const override {
        T alpha = alpha_;
        return input.apply([alpha](T x) { return x > T(0) ? x : alpha * x; });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& /*output*/,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = input[i] > T(0) ? grad_output[i] : alpha_ * grad_output[i];
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::LeakyReLU; }
    std::string name() const override { return "leaky_relu"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<LeakyReLUActivation>(alpha_);
    }

private:
    T alpha_;
};

/**
 * ELU activation: f(x) = x if x > 0, else alpha * (exp(x) - 1)
 */
template<typename T = float>
class ELUActivation : public Activation<T> {
public:
    explicit ELUActivation(T alpha = T(1)) : alpha_(alpha) {}

    Tensor<T> forward(const Tensor<T>& input) const override {
        T alpha = alpha_;
        return input.apply([alpha](T x) {
            return x > T(0) ? x : alpha * (std::exp(x) - T(1));
        });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& output,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            result[i] = input[i] > T(0) ? grad_output[i]
                                        : grad_output[i] * (output[i] + alpha_);
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::ELU; }
    std::string name() const override { return "elu"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<ELUActivation>(alpha_);
    }

private:
    T alpha_;
};

/**
 * SELU activation (self-normalizing).
 */
template<typename T = float>
class SELUActivation : public Activation<T> {
public:
    SELUActivation()
        : alpha_(T(1.6732632423543772848170429916717))
        , scale_(T(1.0507009873554804934193349852946)) {}

    Tensor<T> forward(const Tensor<T>& input) const override {
        T alpha = alpha_;
        T scale = scale_;
        return input.apply([alpha, scale](T x) {
            return scale * (x > T(0) ? x : alpha * (std::exp(x) - T(1)));
        });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& output,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            if (input[i] > T(0)) {
                result[i] = scale_ * grad_output[i];
            } else {
                result[i] = grad_output[i] * (output[i] + scale_ * alpha_);
            }
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::SELU; }
    std::string name() const override { return "selu"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<SELUActivation>();
    }

private:
    T alpha_;
    T scale_;
};

/**
 * Sigmoid activation: f(x) = 1 / (1 + exp(-x))
 */
template<typename T = float>
class SigmoidActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return input.apply([](T x) { return T(1) / (T(1) + std::exp(-x)); });
    }

    Tensor<T> backward(const Tensor<T>& /*input*/,
                       const Tensor<T>& output,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < output.size(); ++i) {
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            result[i] = grad_output[i] * output[i] * (T(1) - output[i]);
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::Sigmoid; }
    std::string name() const override { return "sigmoid"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<SigmoidActivation>();
    }
};

/**
 * Tanh activation: f(x) = tanh(x)
 */
template<typename T = float>
class TanhActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return input.apply([](T x) { return std::tanh(x); });
    }

    Tensor<T> backward(const Tensor<T>& /*input*/,
                       const Tensor<T>& output,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < output.size(); ++i) {
            // tanh'(x) = 1 - tanh(x)^2
            result[i] = grad_output[i] * (T(1) - output[i] * output[i]);
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::Tanh; }
    std::string name() const override { return "tanh"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<TanhActivation>();
    }
};

/**
 * Softmax activation (for classification).
 * Applied row-wise for 2D tensors.
 */
template<typename T = float>
class SoftmaxActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        Tensor<T> result(input.shape());

        if (input.rank() == 1) {
            // 1D case
            T max_val = input.max();
            T sum = T(0);
            for (size_t i = 0; i < input.size(); ++i) {
                result[i] = std::exp(input[i] - max_val);
                sum += result[i];
            }
            for (size_t i = 0; i < input.size(); ++i) {
                result[i] /= sum;
            }
        } else if (input.rank() == 2) {
            // 2D case: apply softmax row-wise
            size_t rows = input.shape()[0];
            size_t cols = input.shape()[1];
            for (size_t i = 0; i < rows; ++i) {
                // Find max for numerical stability
                T max_val = input.at(i, 0);
                for (size_t j = 1; j < cols; ++j) {
                    max_val = std::max(max_val, input.at(i, j));
                }
                // Compute exp and sum
                T sum = T(0);
                for (size_t j = 0; j < cols; ++j) {
                    result.at(i, j) = std::exp(input.at(i, j) - max_val);
                    sum += result.at(i, j);
                }
                // Normalize
                for (size_t j = 0; j < cols; ++j) {
                    result.at(i, j) /= sum;
                }
            }
        }
        return result;
    }

    Tensor<T> backward(const Tensor<T>& /*input*/,
                       const Tensor<T>& output,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());

        if (output.rank() == 1) {
            // Jacobian: diag(s) - s * s^T
            size_t n = output.size();
            for (size_t i = 0; i < n; ++i) {
                T sum = T(0);
                for (size_t j = 0; j < n; ++j) {
                    sum += output[j] * grad_output[j];
                }
                result[i] = output[i] * (grad_output[i] - sum);
            }
        } else if (output.rank() == 2) {
            size_t rows = output.shape()[0];
            size_t cols = output.shape()[1];
            for (size_t i = 0; i < rows; ++i) {
                T sum = T(0);
                for (size_t j = 0; j < cols; ++j) {
                    sum += output.at(i, j) * grad_output.at(i, j);
                }
                for (size_t j = 0; j < cols; ++j) {
                    result.at(i, j) = output.at(i, j) * (grad_output.at(i, j) - sum);
                }
            }
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::Softmax; }
    std::string name() const override { return "softmax"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<SoftmaxActivation>();
    }
};

/**
 * Swish activation: f(x) = x * sigmoid(x)
 */
template<typename T = float>
class SwishActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return input.apply([](T x) {
            T sig = T(1) / (T(1) + std::exp(-x));
            return x * sig;
        });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& output,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            T sig = T(1) / (T(1) + std::exp(-input[i]));
            // swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            //           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            //           = swish(x) + sigmoid(x) * (1 - swish(x))
            result[i] = grad_output[i] * (output[i] + sig * (T(1) - output[i]));
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::Swish; }
    std::string name() const override { return "swish"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<SwishActivation>();
    }
};

/**
 * GELU activation (Gaussian Error Linear Unit).
 * Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
template<typename T = float>
class GELUActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        const T sqrt_2_pi = T(0.7978845608028654);
        const T coeff = T(0.044715);
        return input.apply([sqrt_2_pi, coeff](T x) {
            T inner = sqrt_2_pi * (x + coeff * x * x * x);
            return T(0.5) * x * (T(1) + std::tanh(inner));
        });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& /*output*/,
                       const Tensor<T>& grad_output) const override {
        const T sqrt_2_pi = T(0.7978845608028654);
        const T coeff = T(0.044715);
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            T x = input[i];
            T x3 = x * x * x;
            T inner = sqrt_2_pi * (x + coeff * x3);
            T tanh_inner = std::tanh(inner);
            T sech2 = T(1) - tanh_inner * tanh_inner;

            // Derivative
            T grad = T(0.5) * (T(1) + tanh_inner) +
                     T(0.5) * x * sech2 * sqrt_2_pi * (T(1) + T(3) * coeff * x * x);
            result[i] = grad_output[i] * grad;
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::GELU; }
    std::string name() const override { return "gelu"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<GELUActivation>();
    }
};

/**
 * Softplus activation: f(x) = log(1 + exp(x))
 */
template<typename T = float>
class SoftplusActivation : public Activation<T> {
public:
    Tensor<T> forward(const Tensor<T>& input) const override {
        return input.apply([](T x) {
            // For numerical stability
            if (x > T(20)) return x;
            return std::log(T(1) + std::exp(x));
        });
    }

    Tensor<T> backward(const Tensor<T>& input,
                       const Tensor<T>& /*output*/,
                       const Tensor<T>& grad_output) const override {
        Tensor<T> result(grad_output.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            // softplus'(x) = sigmoid(x)
            result[i] = grad_output[i] / (T(1) + std::exp(-input[i]));
        }
        return result;
    }

    ActivationType type() const override { return ActivationType::Softplus; }
    std::string name() const override { return "softplus"; }
    std::unique_ptr<Activation<T>> clone() const override {
        return std::make_unique<SoftplusActivation>();
    }
};

// Factory implementations
template<typename T>
std::unique_ptr<Activation<T>> Activation<T>::create(ActivationType type) {
    switch (type) {
        case ActivationType::Linear:
            return std::make_unique<LinearActivation<T>>();
        case ActivationType::ReLU:
            return std::make_unique<ReLUActivation<T>>();
        case ActivationType::LeakyReLU:
            return std::make_unique<LeakyReLUActivation<T>>();
        case ActivationType::ELU:
            return std::make_unique<ELUActivation<T>>();
        case ActivationType::SELU:
            return std::make_unique<SELUActivation<T>>();
        case ActivationType::Sigmoid:
            return std::make_unique<SigmoidActivation<T>>();
        case ActivationType::Tanh:
            return std::make_unique<TanhActivation<T>>();
        case ActivationType::Softmax:
            return std::make_unique<SoftmaxActivation<T>>();
        case ActivationType::Swish:
            return std::make_unique<SwishActivation<T>>();
        case ActivationType::GELU:
            return std::make_unique<GELUActivation<T>>();
        case ActivationType::Softplus:
            return std::make_unique<SoftplusActivation<T>>();
        default:
            return std::make_unique<ReLUActivation<T>>();
    }
}

template<typename T>
std::unique_ptr<Activation<T>> Activation<T>::create(const std::string& name) {
    if (name == "linear" || name == "none") return create(ActivationType::Linear);
    if (name == "relu") return create(ActivationType::ReLU);
    if (name == "leaky_relu") return create(ActivationType::LeakyReLU);
    if (name == "elu") return create(ActivationType::ELU);
    if (name == "selu") return create(ActivationType::SELU);
    if (name == "sigmoid") return create(ActivationType::Sigmoid);
    if (name == "tanh") return create(ActivationType::Tanh);
    if (name == "softmax") return create(ActivationType::Softmax);
    if (name == "swish" || name == "silu") return create(ActivationType::Swish);
    if (name == "gelu") return create(ActivationType::GELU);
    if (name == "softplus") return create(ActivationType::Softplus);
    return create(ActivationType::ReLU);  // Default
}

} // namespace core
} // namespace dnn
