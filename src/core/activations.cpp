#include "dnn/core/activations.hpp"

namespace dnn {
namespace core {

// Explicit template instantiations
template class Activation<float>;
template class Activation<double>;
template class ReLUActivation<float>;
template class ReLUActivation<double>;
template class SigmoidActivation<float>;
template class SigmoidActivation<double>;
template class TanhActivation<float>;
template class TanhActivation<double>;
template class SoftmaxActivation<float>;
template class SoftmaxActivation<double>;

} // namespace core
} // namespace dnn
