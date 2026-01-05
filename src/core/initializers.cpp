#include "dnn/core/initializers.hpp"

namespace dnn {
namespace core {

// Explicit template instantiations
template class Initializer<float>;
template class Initializer<double>;
template class ZerosInitializer<float>;
template class ZerosInitializer<double>;
template class HeNormalInitializer<float>;
template class HeNormalInitializer<double>;
template class XavierNormalInitializer<float>;
template class XavierNormalInitializer<double>;

} // namespace core
} // namespace dnn
