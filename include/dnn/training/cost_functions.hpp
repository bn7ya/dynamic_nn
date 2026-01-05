#pragma once

#include "../core/tensor.hpp"
#include <memory>
#include <string>
#include <cmath>
#include <algorithm>

namespace dnn {
namespace training {

using core::Tensor;

/**
 * Available cost function types (user-selectable).
 */
enum class CostFunctionType {
    MeanSquaredError,           // MSE
    MeanAbsoluteError,          // MAE
    CrossEntropy,               // For multi-class classification
    BinaryCrossEntropy,         // For binary classification
    CategoricalCrossEntropy,    // Same as CrossEntropy but explicit
    SparseCategoricalCrossEntropy,  // For integer labels
    HuberLoss,                  // Robust to outliers
    LogCosh,                    // Smooth approximation to MAE
    KLDivergence,              // Kullback-Leibler divergence
    CosineSimilarity           // For embedding tasks
};

/**
 * Base class for cost functions.
 */
template<typename T = float>
class CostFunction {
public:
    virtual ~CostFunction() = default;

    /**
     * Compute the cost/loss.
     * @param predicted Network output
     * @param actual Target values
     * @return Scalar cost value
     */
    virtual T compute(const Tensor<T>& predicted,
                      const Tensor<T>& actual) const = 0;

    /**
     * Compute gradient of cost with respect to predicted.
     */
    virtual Tensor<T> gradient(const Tensor<T>& predicted,
                               const Tensor<T>& actual) const = 0;

    /**
     * Get the name of this cost function.
     */
    virtual std::string name() const = 0;

    /**
     * Get the type.
     */
    virtual CostFunctionType type() const = 0;

    /**
     * Clone the cost function.
     */
    virtual std::unique_ptr<CostFunction> clone() const = 0;

    /**
     * Factory method.
     */
    static std::unique_ptr<CostFunction> create(CostFunctionType type);
    static std::unique_ptr<CostFunction> create(const std::string& name);
};

/**
 * Mean Squared Error: L = (1/n) * sum((predicted - actual)^2)
 */
template<typename T = float>
class MSECost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - actual[i];
            sum += diff * diff;
        }
        return sum / static_cast<T>(predicted.size());
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        T scale = T(2) / static_cast<T>(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            grad[i] = scale * (predicted[i] - actual[i]);
        }
        return grad;
    }

    std::string name() const override { return "MSE"; }
    CostFunctionType type() const override { return CostFunctionType::MeanSquaredError; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<MSECost>();
    }
};

/**
 * Mean Absolute Error: L = (1/n) * sum(|predicted - actual|)
 */
template<typename T = float>
class MAECost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        for (size_t i = 0; i < predicted.size(); ++i) {
            sum += std::abs(predicted[i] - actual[i]);
        }
        return sum / static_cast<T>(predicted.size());
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        T scale = T(1) / static_cast<T>(predicted.size());
        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - actual[i];
            grad[i] = scale * (diff > T(0) ? T(1) : (diff < T(0) ? T(-1) : T(0)));
        }
        return grad;
    }

    std::string name() const override { return "MAE"; }
    CostFunctionType type() const override { return CostFunctionType::MeanAbsoluteError; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<MAECost>();
    }
};

/**
 * Cross-Entropy Loss: L = -sum(actual * log(predicted))
 * Assumes predicted is after softmax (probabilities).
 */
template<typename T = float>
class CrossEntropyCost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        const T epsilon = T(1e-7);

        for (size_t i = 0; i < predicted.size(); ++i) {
            if (actual[i] > T(0)) {
                T p = std::max(epsilon, std::min(T(1) - epsilon, predicted[i]));
                sum -= actual[i] * std::log(p);
            }
        }

        // If batch, average over samples
        if (predicted.rank() == 2) {
            return sum / static_cast<T>(predicted.shape()[0]);
        }
        return sum;
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        const T epsilon = T(1e-7);

        if (predicted.rank() == 1) {
            for (size_t i = 0; i < predicted.size(); ++i) {
                T p = std::max(epsilon, std::min(T(1) - epsilon, predicted[i]));
                grad[i] = -actual[i] / p;
            }
        } else if (predicted.rank() == 2) {
            size_t batch_size = predicted.shape()[0];
            size_t classes = predicted.shape()[1];
            for (size_t b = 0; b < batch_size; ++b) {
                for (size_t c = 0; c < classes; ++c) {
                    T p = std::max(epsilon, std::min(T(1) - epsilon, predicted.at(b, c)));
                    grad.at(b, c) = -actual.at(b, c) / (p * static_cast<T>(batch_size));
                }
            }
        }
        return grad;
    }

    std::string name() const override { return "CrossEntropy"; }
    CostFunctionType type() const override { return CostFunctionType::CrossEntropy; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<CrossEntropyCost>();
    }
};

/**
 * Binary Cross-Entropy: L = -[y*log(p) + (1-y)*log(1-p)]
 */
template<typename T = float>
class BinaryCrossEntropyCost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        const T epsilon = T(1e-7);

        for (size_t i = 0; i < predicted.size(); ++i) {
            T p = std::max(epsilon, std::min(T(1) - epsilon, predicted[i]));
            T y = actual[i];
            sum -= y * std::log(p) + (T(1) - y) * std::log(T(1) - p);
        }
        return sum / static_cast<T>(predicted.size());
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        const T epsilon = T(1e-7);
        T scale = T(1) / static_cast<T>(predicted.size());

        for (size_t i = 0; i < predicted.size(); ++i) {
            T p = std::max(epsilon, std::min(T(1) - epsilon, predicted[i]));
            T y = actual[i];
            grad[i] = scale * (-(y / p) + (T(1) - y) / (T(1) - p));
        }
        return grad;
    }

    std::string name() const override { return "BinaryCrossEntropy"; }
    CostFunctionType type() const override { return CostFunctionType::BinaryCrossEntropy; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<BinaryCrossEntropyCost>();
    }
};

/**
 * Huber Loss: smooth approximation between MSE and MAE.
 * L = 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
 */
template<typename T = float>
class HuberCost : public CostFunction<T> {
public:
    explicit HuberCost(T delta = T(1)) : delta_(delta) {}

    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - actual[i];
            T abs_diff = std::abs(diff);
            if (abs_diff <= delta_) {
                sum += T(0.5) * diff * diff;
            } else {
                sum += delta_ * (abs_diff - T(0.5) * delta_);
            }
        }
        return sum / static_cast<T>(predicted.size());
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        T scale = T(1) / static_cast<T>(predicted.size());

        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - actual[i];
            T abs_diff = std::abs(diff);
            if (abs_diff <= delta_) {
                grad[i] = scale * diff;
            } else {
                grad[i] = scale * delta_ * (diff > T(0) ? T(1) : T(-1));
            }
        }
        return grad;
    }

    std::string name() const override { return "Huber"; }
    CostFunctionType type() const override { return CostFunctionType::HuberLoss; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<HuberCost>(delta_);
    }

private:
    T delta_;
};

/**
 * Log-Cosh Loss: log(cosh(predicted - actual))
 * Smooth approximation to MAE.
 */
template<typename T = float>
class LogCoshCost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - actual[i];
            sum += std::log(std::cosh(diff));
        }
        return sum / static_cast<T>(predicted.size());
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        T scale = T(1) / static_cast<T>(predicted.size());

        for (size_t i = 0; i < predicted.size(); ++i) {
            T diff = predicted[i] - actual[i];
            grad[i] = scale * std::tanh(diff);
        }
        return grad;
    }

    std::string name() const override { return "LogCosh"; }
    CostFunctionType type() const override { return CostFunctionType::LogCosh; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<LogCoshCost>();
    }
};

/**
 * KL Divergence: sum(actual * log(actual / predicted))
 */
template<typename T = float>
class KLDivergenceCost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T sum = T(0);
        const T epsilon = T(1e-7);

        for (size_t i = 0; i < predicted.size(); ++i) {
            if (actual[i] > T(0)) {
                T p = std::max(epsilon, predicted[i]);
                T q = actual[i];
                sum += q * std::log(q / p);
            }
        }
        return sum;
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        const T epsilon = T(1e-7);

        for (size_t i = 0; i < predicted.size(); ++i) {
            T p = std::max(epsilon, predicted[i]);
            grad[i] = -actual[i] / p;
        }
        return grad;
    }

    std::string name() const override { return "KLDivergence"; }
    CostFunctionType type() const override { return CostFunctionType::KLDivergence; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<KLDivergenceCost>();
    }
};

/**
 * Cosine Similarity Loss: 1 - cos_similarity(predicted, actual)
 */
template<typename T = float>
class CosineSimilarityCost : public CostFunction<T> {
public:
    T compute(const Tensor<T>& predicted,
              const Tensor<T>& actual) const override {
        T dot_product = T(0);
        T norm_pred = T(0);
        T norm_actual = T(0);

        for (size_t i = 0; i < predicted.size(); ++i) {
            dot_product += predicted[i] * actual[i];
            norm_pred += predicted[i] * predicted[i];
            norm_actual += actual[i] * actual[i];
        }

        norm_pred = std::sqrt(norm_pred);
        norm_actual = std::sqrt(norm_actual);

        const T epsilon = T(1e-7);
        T similarity = dot_product / (norm_pred * norm_actual + epsilon);

        return T(1) - similarity;
    }

    Tensor<T> gradient(const Tensor<T>& predicted,
                       const Tensor<T>& actual) const override {
        Tensor<T> grad(predicted.shape());
        const T epsilon = T(1e-7);

        T dot_product = T(0);
        T norm_pred = T(0);
        T norm_actual = T(0);

        for (size_t i = 0; i < predicted.size(); ++i) {
            dot_product += predicted[i] * actual[i];
            norm_pred += predicted[i] * predicted[i];
            norm_actual += actual[i] * actual[i];
        }

        norm_pred = std::sqrt(norm_pred);
        norm_actual = std::sqrt(norm_actual);

        T denom = norm_pred * norm_actual + epsilon;

        for (size_t i = 0; i < predicted.size(); ++i) {
            // d/dp_i (1 - dot/(||p||*||a||))
            // = -a_i/(||p||*||a||) + dot*p_i/(||p||^3*||a||)
            grad[i] = -(actual[i] / denom) +
                      (dot_product * predicted[i]) / (norm_pred * norm_pred * denom);
        }
        return grad;
    }

    std::string name() const override { return "CosineSimilarity"; }
    CostFunctionType type() const override { return CostFunctionType::CosineSimilarity; }
    std::unique_ptr<CostFunction<T>> clone() const override {
        return std::make_unique<CosineSimilarityCost>();
    }
};

// Factory implementations
template<typename T>
std::unique_ptr<CostFunction<T>> CostFunction<T>::create(CostFunctionType type) {
    switch (type) {
        case CostFunctionType::MeanSquaredError:
            return std::make_unique<MSECost<T>>();
        case CostFunctionType::MeanAbsoluteError:
            return std::make_unique<MAECost<T>>();
        case CostFunctionType::CrossEntropy:
        case CostFunctionType::CategoricalCrossEntropy:
        case CostFunctionType::SparseCategoricalCrossEntropy:
            return std::make_unique<CrossEntropyCost<T>>();
        case CostFunctionType::BinaryCrossEntropy:
            return std::make_unique<BinaryCrossEntropyCost<T>>();
        case CostFunctionType::HuberLoss:
            return std::make_unique<HuberCost<T>>();
        case CostFunctionType::LogCosh:
            return std::make_unique<LogCoshCost<T>>();
        case CostFunctionType::KLDivergence:
            return std::make_unique<KLDivergenceCost<T>>();
        case CostFunctionType::CosineSimilarity:
            return std::make_unique<CosineSimilarityCost<T>>();
        default:
            return std::make_unique<MSECost<T>>();
    }
}

template<typename T>
std::unique_ptr<CostFunction<T>> CostFunction<T>::create(const std::string& name) {
    if (name == "mse" || name == "MSE" || name == "mean_squared_error") {
        return create(CostFunctionType::MeanSquaredError);
    }
    if (name == "mae" || name == "MAE" || name == "mean_absolute_error") {
        return create(CostFunctionType::MeanAbsoluteError);
    }
    if (name == "cross_entropy" || name == "CrossEntropy" || name == "categorical_crossentropy") {
        return create(CostFunctionType::CrossEntropy);
    }
    if (name == "binary_cross_entropy" || name == "BinaryCrossEntropy") {
        return create(CostFunctionType::BinaryCrossEntropy);
    }
    if (name == "huber" || name == "Huber") {
        return create(CostFunctionType::HuberLoss);
    }
    if (name == "log_cosh" || name == "LogCosh") {
        return create(CostFunctionType::LogCosh);
    }
    if (name == "kl_divergence" || name == "KLDivergence") {
        return create(CostFunctionType::KLDivergence);
    }
    if (name == "cosine_similarity" || name == "CosineSimilarity") {
        return create(CostFunctionType::CosineSimilarity);
    }
    return create(CostFunctionType::MeanSquaredError);  // Default
}

} // namespace training
} // namespace dnn
