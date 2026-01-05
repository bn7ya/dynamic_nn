// Tensor implementation
// Most functionality is in the header due to templates

#include "dnn/core/tensor.hpp"

namespace dnn {
namespace core {

// Explicit template instantiations
template class Tensor<float>;
template class Tensor<double>;

} // namespace core
} // namespace dnn
