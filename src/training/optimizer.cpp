// optimizer.cpp - Optimizer implementations
// Provides SGD, SGD with Momentum, Adam, and RMSprop optimizers
// with full algorithm parity to Python implementation

#include "dnn/training/optimizer.hpp"

namespace dnn {
namespace training {

// Explicit template instantiations for float
template class Optimizer<float>;
template class SGDOptimizer<float>;
template class SGDMomentumOptimizer<float>;
template class AdamOptimizer<float>;
template class RMSpropOptimizer<float>;

// Explicit template instantiations for double
template class Optimizer<double>;
template class SGDOptimizer<double>;
template class SGDMomentumOptimizer<double>;
template class AdamOptimizer<double>;
template class RMSpropOptimizer<double>;

// Factory method explicit instantiation
template std::unique_ptr<Optimizer<float>> Optimizer<float>::create(const OptimizerConfig& config);
template std::unique_ptr<Optimizer<double>> Optimizer<double>::create(const OptimizerConfig& config);

} // namespace training
} // namespace dnn
