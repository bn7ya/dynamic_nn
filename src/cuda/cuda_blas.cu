/**
 * CUDA BLAS Implementation
 *
 * Implements matrix operations using cuBLAS library.
 * Provides GEMM, GEMV, and dot product operations.
 */

#ifdef DNN_ENABLE_CUDA

#include "dnn/cuda/cuda_common.hpp"
#include "dnn/cuda/cuda_ops.hpp"
#include "dnn/cuda/cuda_tensor.hpp"
#include <cublas_v2.h>

namespace dnn {
namespace cuda {

// ============================================================================
// cuBLAS Handle Management
// ============================================================================

/**
 * Singleton cuBLAS context manager.
 * Ensures proper handle lifecycle and provides thread-safe access.
 */
class CublasContext {
public:
    static cublasHandle_t& handle() {
        static CublasContext instance;
        return instance.handle_;
    }

private:
    CublasContext() {
        CUBLAS_CHECK(cublasCreate(&handle_));
        // Use default math mode (enables tensor cores when available)
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH));
    }

    ~CublasContext() {
        if (handle_) {
            cublasDestroy(handle_);
        }
    }

    CublasContext(const CublasContext&) = delete;
    CublasContext& operator=(const CublasContext&) = delete;

    cublasHandle_t handle_;
};

// ============================================================================
// CublasHandle wrapper implementation
// ============================================================================

CublasHandle::CublasHandle() : handle(nullptr) {
    if (is_cuda_available()) {
        cublasHandle_t h;
        if (cublasCreate(&h) == CUBLAS_STATUS_SUCCESS) {
            handle = h;
        }
    }
}

CublasHandle::~CublasHandle() {
    if (handle) {
        cublasDestroy(static_cast<cublasHandle_t>(handle));
    }
}

// ============================================================================
// GEMM - General Matrix Multiplication
// ============================================================================

/**
 * Single precision GEMM: C = alpha * A @ B + beta * C
 *
 * Note: cuBLAS uses column-major ordering. For row-major tensors,
 * we compute C^T = B^T @ A^T, which gives us C in row-major order.
 */
template<>
void cuda_gemm<float>(const CudaTensor<float>& A, const CudaTensor<float>& B,
                      CudaTensor<float>& C,
                      float alpha, float beta,
                      bool transpose_A, bool transpose_B) {
    // Row-major @ row-major using col-major cuBLAS via the standard trick:
    //   C_rm = op_a(A_rm) @ op_b(B_rm)  ->  C_cm = op_b(B_rm)^T @ op_a(A_rm)^T
    // Row-major bytes interpreted as col-major are the TRANSPOSE of the
    // row-major matrix. So:
    //   - To make cuBLAS see op_a(A_rm)^T from A_rm bytes:
    //       transpose_A=false: want A_rm^T -> bytes already give A_rm^T -> op=N
    //       transpose_A=true:  want A_rm   -> bytes give A_rm^T,  un-transpose -> op=T
    //   - Same for the B operand with transpose_B.
    // The cuBLAS A slot receives our B_rm pointer; the cuBLAS B slot
    // receives our A_rm pointer (operand swap, see formula above).
    size_t m = transpose_A ? A.shape()[1] : A.shape()[0];
    size_t k = transpose_A ? A.shape()[0] : A.shape()[1];
    size_t n = transpose_B ? B.shape()[0] : B.shape()[1];

    int lda = static_cast<int>(A.shape()[1]);  // row-major row stride of A
    int ldb = static_cast<int>(B.shape()[1]);  // row-major row stride of B
    int ldc = static_cast<int>(C.shape()[1]);  // row-major row stride of C

    cublasOperation_t op_for_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_for_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasSgemm(
        CublasContext::handle(),
        op_for_B, op_for_A,
        static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha,
        static_cast<const float*>(B.device_data()), ldb,
        static_cast<const float*>(A.device_data()), lda,
        &beta,
        static_cast<float*>(C.device_data()), ldc
    ));
}

/**
 * Double precision GEMM: C = alpha * A @ B + beta * C
 */
template<>
void cuda_gemm<double>(const CudaTensor<double>& A, const CudaTensor<double>& B,
                       CudaTensor<double>& C,
                       double alpha, double beta,
                       bool transpose_A, bool transpose_B) {
    size_t m = transpose_A ? A.shape()[1] : A.shape()[0];
    size_t k = transpose_A ? A.shape()[0] : A.shape()[1];
    size_t n = transpose_B ? B.shape()[0] : B.shape()[1];

    int lda = static_cast<int>(A.shape()[1]);
    int ldb = static_cast<int>(B.shape()[1]);
    int ldc = static_cast<int>(C.shape()[1]);

    // See cuda_gemm<float> for the row-major-via-col-major derivation.
    cublasOperation_t op_for_A = transpose_A ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_for_B = transpose_B ? CUBLAS_OP_T : CUBLAS_OP_N;

    CUBLAS_CHECK(cublasDgemm(
        CublasContext::handle(),
        op_for_B, op_for_A,
        static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha,
        static_cast<const double*>(B.device_data()), ldb,
        static_cast<const double*>(A.device_data()), lda,
        &beta,
        static_cast<double*>(C.device_data()), ldc
    ));
}

// ============================================================================
// GEMV - General Matrix-Vector Multiplication
// ============================================================================

/**
 * Single precision GEMV: y = alpha * A @ x + beta * y
 */
template<>
void cuda_gemv<float>(const CudaTensor<float>& A, const CudaTensor<float>& x,
                      CudaTensor<float>& y,
                      float alpha, float beta, bool transpose) {
    size_t m = A.shape()[0];
    size_t n = A.shape()[1];
    int lda = static_cast<int>(n);

    // For row-major matrix, we need to transpose the operation
    cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;

    CUBLAS_CHECK(cublasSgemv(
        CublasContext::handle(),
        op,
        static_cast<int>(n), static_cast<int>(m),
        &alpha,
        static_cast<const float*>(A.device_data()), lda,
        static_cast<const float*>(x.device_data()), 1,
        &beta,
        static_cast<float*>(y.device_data()), 1
    ));
}

/**
 * Double precision GEMV: y = alpha * A @ x + beta * y
 */
template<>
void cuda_gemv<double>(const CudaTensor<double>& A, const CudaTensor<double>& x,
                       CudaTensor<double>& y,
                       double alpha, double beta, bool transpose) {
    size_t m = A.shape()[0];
    size_t n = A.shape()[1];
    int lda = static_cast<int>(n);

    cublasOperation_t op = transpose ? CUBLAS_OP_N : CUBLAS_OP_T;

    CUBLAS_CHECK(cublasDgemv(
        CublasContext::handle(),
        op,
        static_cast<int>(n), static_cast<int>(m),
        &alpha,
        static_cast<const double*>(A.device_data()), lda,
        static_cast<const double*>(x.device_data()), 1,
        &beta,
        static_cast<double*>(y.device_data()), 1
    ));
}

// ============================================================================
// Dot Product
// ============================================================================

/**
 * Single precision dot product: result = x . y
 */
template<>
float cuda_dot<float>(const CudaTensor<float>& x, const CudaTensor<float>& y) {
    float result = 0.0f;

    CUBLAS_CHECK(cublasSdot(
        CublasContext::handle(),
        static_cast<int>(x.size()),
        static_cast<const float*>(x.device_data()), 1,
        static_cast<const float*>(y.device_data()), 1,
        &result
    ));

    return result;
}

/**
 * Double precision dot product: result = x . y
 */
template<>
double cuda_dot<double>(const CudaTensor<double>& x, const CudaTensor<double>& y) {
    double result = 0.0;

    CUBLAS_CHECK(cublasDdot(
        CublasContext::handle(),
        static_cast<int>(x.size()),
        static_cast<const double*>(x.device_data()), 1,
        static_cast<const double*>(y.device_data()), 1,
        &result
    ));

    return result;
}

// ============================================================================
// Copy Operation
// ============================================================================

template<>
void cuda_copy<float>(const CudaTensor<float>& src, CudaTensor<float>& dst) {
    if (src.size() != dst.size()) {
        throw std::runtime_error("cuda_copy: size mismatch");
    }

    CUBLAS_CHECK(cublasScopy(
        CublasContext::handle(),
        static_cast<int>(src.size()),
        static_cast<const float*>(src.device_data()), 1,
        static_cast<float*>(dst.device_data()), 1
    ));
}

template<>
void cuda_copy<double>(const CudaTensor<double>& src, CudaTensor<double>& dst) {
    if (src.size() != dst.size()) {
        throw std::runtime_error("cuda_copy: size mismatch");
    }

    CUBLAS_CHECK(cublasDcopy(
        CublasContext::handle(),
        static_cast<int>(src.size()),
        static_cast<const double*>(src.device_data()), 1,
        static_cast<double*>(dst.device_data()), 1
    ));
}

} // namespace cuda
} // namespace dnn

#endif // DNN_ENABLE_CUDA
