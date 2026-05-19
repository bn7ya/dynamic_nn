#pragma once

#include "../memory/aligned_allocator.hpp"
#include "../memory/resource_monitor.hpp"
#include "../simd/simd_ops.hpp"
#include "../exceptions/dnn_exception.hpp"
#include <vector>
#include <memory>
#include <numeric>
#include <cstring>
#include <cassert>
#include <initializer_list>
#include <functional>
#include <sstream>
#include <cmath>

namespace dnn {
namespace core {

/**
 * Multi-dimensional tensor with SIMD-aligned storage.
 * Template supports float and double precision.
 */
template<typename T = float>
class Tensor {
    static_assert(std::is_floating_point_v<T>, "Tensor only supports floating point types");

public:
    using value_type = T;
    using size_type = size_t;
    using shape_type = std::vector<size_t>;

    /**
     * Immutable, non-owning view over a Tensor's storage. Exposes only
     * const accessors so a const Tensor can be aliased without a
     * const_cast and without any way to mutate through the alias.
     */
    class ConstView {
    public:
        ConstView(const T* data, shape_type shape, shape_type strides,
                  size_t total_size)
            : data_(data), shape_(std::move(shape)),
              strides_(std::move(strides)), total_size_(total_size) {}

        const shape_type& shape() const { return shape_; }
        size_t rank() const { return shape_.size(); }
        size_t size() const { return total_size_; }
        const T* data() const { return data_; }
        const T& operator[](size_t i) const { return data_[i]; }
        const T& at(size_t i, size_t j) const {
            return data_[i * strides_[0] + j];
        }

    private:
        const T* data_;
        shape_type shape_;
        shape_type strides_;
        size_t total_size_;
    };

    /**
     * Default constructor - creates empty tensor.
     */
    Tensor() : data_(nullptr), total_size_(0), owns_data_(true) {}

    /**
     * Construct tensor with given shape.
     */
    explicit Tensor(const shape_type& shape)
        : shape_(shape)
        , owns_data_(true) {
        compute_strides();
        allocate();
        fill(T(0));
    }

    /**
     * Construct tensor with given shape and initial value.
     */
    Tensor(const shape_type& shape, T init_value)
        : shape_(shape)
        , owns_data_(true) {
        compute_strides();
        allocate();
        fill(init_value);
    }

    /**
     * Construct 1D tensor from initializer list.
     */
    Tensor(std::initializer_list<T> data)
        : shape_({data.size()})
        , owns_data_(true) {
        compute_strides();
        allocate();
        std::copy(data.begin(), data.end(), data_);
    }

    /**
     * Copy constructor.
     */
    Tensor(const Tensor& other)
        : shape_(other.shape_)
        , strides_(other.strides_)
        , total_size_(other.total_size_)
        , owns_data_(true) {
        if (other.data_ && total_size_ > 0) {
            allocate();
            std::memcpy(data_, other.data_, total_size_ * sizeof(T));
        }
    }

    /**
     * Move constructor.
     */
    Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_))
        , strides_(std::move(other.strides_))
        , data_(other.data_)
        , total_size_(other.total_size_)
        , owns_data_(other.owns_data_) {
        other.data_ = nullptr;
        other.total_size_ = 0;
        other.owns_data_ = false;
    }

    /**
     * Copy assignment.
     */
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            deallocate();
            shape_ = other.shape_;
            strides_ = other.strides_;
            total_size_ = other.total_size_;
            owns_data_ = true;
            if (other.data_ && total_size_ > 0) {
                allocate();
                std::memcpy(data_, other.data_, total_size_ * sizeof(T));
            }
        }
        return *this;
    }

    /**
     * Move assignment.
     */
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            deallocate();
            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            data_ = other.data_;
            total_size_ = other.total_size_;
            owns_data_ = other.owns_data_;
            other.data_ = nullptr;
            other.total_size_ = 0;
            other.owns_data_ = false;
        }
        return *this;
    }

    /**
     * Destructor.
     */
    ~Tensor() {
        deallocate();
    }

    // Shape and dimension access
    const shape_type& shape() const { return shape_; }
    size_t rank() const { return shape_.size(); }
    size_t size() const { return total_size_; }
    size_t size(size_t dim) const { return shape_[dim]; }
    const shape_type& strides() const { return strides_; }
    bool empty() const { return total_size_ == 0; }

    // Data access
    T* data() { return data_; }
    const T* data() const { return data_; }

    /**
     * Element access with multi-dimensional indices.
     */
    T& at(const std::vector<size_t>& indices) {
        return data_[compute_offset(indices)];
    }

    const T& at(const std::vector<size_t>& indices) const {
        return data_[compute_offset(indices)];
    }

    /**
     * Linear element access.
     */
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }

    /**
     * 2D access (for matrices).
     */
    T& at(size_t i, size_t j) {
        assert(rank() == 2 && i < shape_[0] && j < shape_[1] &&
               "Tensor::at(i,j) index out of bounds");
        return data_[i * strides_[0] + j];
    }

    const T& at(size_t i, size_t j) const {
        assert(rank() == 2 && i < shape_[0] && j < shape_[1] &&
               "Tensor::at(i,j) index out of bounds");
        return data_[i * strides_[0] + j];
    }

    /**
     * Reshape tensor (must preserve total size).
     */
    void reshape(const shape_type& new_shape) {
        size_t new_size = std::accumulate(new_shape.begin(), new_shape.end(),
                                          size_t(1), std::multiplies<size_t>());
        if (new_size != total_size_) {
            std::ostringstream oss;
            oss << "Cannot reshape from size " << total_size_ << " to " << new_size;
            throw exceptions::ShapeException("reshape", oss.str());
        }
        shape_ = new_shape;
        compute_strides();
    }

    /**
     * Fill tensor with value.
     */
    void fill(T value) {
        if (!data_ || total_size_ == 0) return;
        simd::ScalarOps<T> ops;
        ops.fill(data_, value, total_size_);
    }

    /**
     * Create a deep copy.
     */
    Tensor clone() const {
        return Tensor(*this);
    }

    /**
     * Create a mutable, non-owning view of this tensor. Non-const: a
     * const Tensor cannot produce a mutable alias of itself. Use
     * const_view() for read-only aliasing of a const tensor.
     */
    Tensor view() {
        Tensor result;
        result.shape_ = shape_;
        result.strides_ = strides_;
        result.data_ = data_;
        result.total_size_ = total_size_;
        result.owns_data_ = false;
        return result;
    }

    /**
     * Read-only, non-owning view. Genuinely cannot mutate the backing
     * storage (no non-const data()/at()), so it is safe to obtain from
     * a const Tensor without const_cast.
     */
    ConstView const_view() const {
        return ConstView(data_, shape_, strides_, total_size_);
    }

    // Arithmetic operations (create new tensors)
    Tensor operator+(const Tensor& other) const {
        check_shape_match(other, "addition");
        Tensor result(shape_);
        simd::ScalarOps<T> ops;
        ops.add(data_, other.data_, result.data_, total_size_);
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        check_shape_match(other, "subtraction");
        Tensor result(shape_);
        simd::ScalarOps<T> ops;
        ops.sub(data_, other.data_, result.data_, total_size_);
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        check_shape_match(other, "element-wise multiplication");
        Tensor result(shape_);
        simd::ScalarOps<T> ops;
        ops.mul(data_, other.data_, result.data_, total_size_);
        return result;
    }

    Tensor operator/(const Tensor& other) const {
        check_shape_match(other, "element-wise division");
        Tensor result(shape_);
        simd::ScalarOps<T> ops;
        ops.div(data_, other.data_, result.data_, total_size_);
        return result;
    }

    Tensor operator+(T scalar) const {
        Tensor result(shape_);
        simd::ScalarOps<T> ops;
        ops.add_scalar(data_, scalar, result.data_, total_size_);
        return result;
    }

    Tensor operator*(T scalar) const {
        Tensor result(shape_);
        simd::ScalarOps<T> ops;
        ops.mul_scalar(data_, scalar, result.data_, total_size_);
        return result;
    }

    // In-place arithmetic
    Tensor& operator+=(const Tensor& other) {
        check_shape_match(other, "addition");
        simd::ScalarOps<T> ops;
        ops.add(data_, other.data_, data_, total_size_);
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        check_shape_match(other, "subtraction");
        simd::ScalarOps<T> ops;
        ops.sub(data_, other.data_, data_, total_size_);
        return *this;
    }

    Tensor& operator*=(T scalar) {
        simd::ScalarOps<T> ops;
        ops.mul_scalar(data_, scalar, data_, total_size_);
        return *this;
    }

    // Reductions
    T sum() const {
        simd::ScalarOps<T> ops;
        return ops.sum(data_, total_size_);
    }

    T max() const {
        simd::ScalarOps<T> ops;
        return ops.max(data_, total_size_);
    }

    T min() const {
        simd::ScalarOps<T> ops;
        return ops.min(data_, total_size_);
    }

    T mean() const {
        return sum() / static_cast<T>(total_size_);
    }

    T variance() const {
        T m = mean();
        T var = T(0);
        for (size_t i = 0; i < total_size_; ++i) {
            T diff = data_[i] - m;
            var += diff * diff;
        }
        return var / static_cast<T>(total_size_);
    }

    T std_dev() const {
        return std::sqrt(variance());
    }

    /**
     * Dot product (for 1D tensors).
     */
    T dot(const Tensor& other) const {
        if (rank() != 1 || other.rank() != 1 || total_size_ != other.total_size_) {
            throw exceptions::ShapeException("dot", "Both tensors must be 1D with same size");
        }
        simd::ScalarOps<T> ops;
        return ops.dot(data_, other.data_, total_size_);
    }

    /**
     * Matrix multiplication (for 2D tensors).
     */
    Tensor matmul(const Tensor& other) const {
        if (rank() != 2 || other.rank() != 2) {
            throw exceptions::ShapeException("matmul", "Both tensors must be 2D");
        }
        if (shape_[1] != other.shape_[0]) {
            std::ostringstream oss;
            oss << "Incompatible shapes: (" << shape_[0] << "," << shape_[1]
                << ") @ (" << other.shape_[0] << "," << other.shape_[1] << ")";
            throw exceptions::ShapeException("matmul", oss.str());
        }

        size_t m = shape_[0];
        size_t k = shape_[1];
        size_t n = other.shape_[1];

        Tensor result(shape_type{m, n});
        simd::ScalarOps<T> ops;
        ops.matmul(data_, other.data_, result.data_,
                   m, k, n, k, n, n);
        return result;
    }

    /**
     * Transpose (for 2D tensors).
     */
    Tensor transpose() const {
        if (rank() != 2) {
            throw exceptions::ShapeException("transpose", "Tensor must be 2D");
        }

        const size_t rows = shape_[0];
        const size_t cols = shape_[1];
        Tensor result(shape_type{cols, rows});

        // Cache-blocked transpose: process TILE x TILE sub-blocks so the
        // strided writes into `result` stay within cache instead of
        // jumping `rows` elements on every inner step.
        constexpr size_t TILE = 32;
        const size_t src_stride = strides_[0];
        const T* src = data_;
        T* dst = result.data_;
        for (size_t ii = 0; ii < rows; ii += TILE) {
            const size_t i_end = std::min(ii + TILE, rows);
            for (size_t jj = 0; jj < cols; jj += TILE) {
                const size_t j_end = std::min(jj + TILE, cols);
                for (size_t i = ii; i < i_end; ++i) {
                    const T* src_row = src + i * src_stride;
                    for (size_t j = jj; j < j_end; ++j) {
                        dst[j * rows + i] = src_row[j];
                    }
                }
            }
        }
        return result;
    }

    /**
     * Apply element-wise function.
     */
    Tensor apply(std::function<T(T)> func) const {
        Tensor result(shape_);
        for (size_t i = 0; i < total_size_; ++i) {
            result.data_[i] = func(data_[i]);
        }
        return result;
    }

    /**
     * Apply element-wise function in-place.
     */
    Tensor& apply_inplace(std::function<T(T)> func) {
        for (size_t i = 0; i < total_size_; ++i) {
            data_[i] = func(data_[i]);
        }
        return *this;
    }

    /**
     * Flatten to 1D.
     */
    Tensor flatten() const {
        Tensor result = clone();
        result.reshape({total_size_});
        return result;
    }

    /**
     * Get row from 2D tensor.
     */
    Tensor row(size_t i) const {
        if (rank() != 2) {
            throw exceptions::ShapeException("row", "Tensor must be 2D");
        }
        Tensor result(shape_type{shape_[1]});
        for (size_t j = 0; j < shape_[1]; ++j) {
            result[j] = at(i, j);
        }
        return result;
    }

    /**
     * Get column from 2D tensor.
     */
    Tensor col(size_t j) const {
        if (rank() != 2) {
            throw exceptions::ShapeException("col", "Tensor must be 2D");
        }
        Tensor result(shape_type{shape_[0]});
        for (size_t i = 0; i < shape_[0]; ++i) {
            result[i] = at(i, j);
        }
        return result;
    }

    /**
     * Check if tensor contains NaN values.
     */
    bool has_nan() const {
        for (size_t i = 0; i < total_size_; ++i) {
            if (std::isnan(data_[i])) return true;
        }
        return false;
    }

    /**
     * Check if tensor contains infinite values.
     */
    bool has_inf() const {
        for (size_t i = 0; i < total_size_; ++i) {
            if (std::isinf(data_[i])) return true;
        }
        return false;
    }

    /**
     * Clip values to range.
     */
    Tensor& clip(T min_val, T max_val) {
        for (size_t i = 0; i < total_size_; ++i) {
            if (data_[i] < min_val) data_[i] = min_val;
            else if (data_[i] > max_val) data_[i] = max_val;
        }
        return *this;
    }

    // Static factory methods
    static Tensor zeros(const shape_type& shape) {
        return Tensor(shape, T(0));
    }

    static Tensor ones(const shape_type& shape) {
        return Tensor(shape, T(1));
    }

    static Tensor eye(size_t n) {
        Tensor result(shape_type{n, n}, T(0));
        for (size_t i = 0; i < n; ++i) {
            result.at(i, i) = T(1);
        }
        return result;
    }

private:
    void compute_strides() {
        strides_.resize(shape_.size());
        if (shape_.empty()) {
            total_size_ = 0;
            return;
        }

        total_size_ = 1;
        for (size_t i = shape_.size(); i > 0; --i) {
            strides_[i - 1] = total_size_;
            total_size_ *= shape_[i - 1];
        }
    }

    void allocate() {
        if (total_size_ == 0) {
            data_ = nullptr;
            return;
        }

        memory::ResourceMonitor::instance().check_can_allocate_tensor(total_size_ * sizeof(T));
        data_ = static_cast<T*>(memory::aligned_alloc(total_size_ * sizeof(T)));
        if (!data_) {
            throw exceptions::MemoryException(total_size_ * sizeof(T), 0, "Tensor allocation");
        }
        memory::ResourceMonitor::instance().register_tensor(total_size_ * sizeof(T));
    }

    void deallocate() {
        if (data_ && owns_data_) {
            memory::ResourceMonitor::instance().unregister_tensor(total_size_ * sizeof(T));
            memory::aligned_free(data_);
        }
        data_ = nullptr;
    }

    size_t compute_offset(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw exceptions::ShapeException("indexing",
                "Number of indices doesn't match tensor rank");
        }
        size_t offset = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw exceptions::InvalidArgumentException("index",
                    "Index out of bounds");
            }
            offset += indices[i] * strides_[i];
        }
        return offset;
    }

    void check_shape_match(const Tensor& other, const std::string& operation) const {
        if (shape_ != other.shape_) {
            std::ostringstream oss;
            oss << "Shape mismatch for " << operation;
            throw exceptions::ShapeException(operation, oss.str());
        }
    }

    shape_type shape_;
    shape_type strides_;
    T* data_ = nullptr;
    size_t total_size_ = 0;
    bool owns_data_ = true;
};

// Type aliases
using TensorF = Tensor<float>;
using TensorD = Tensor<double>;

} // namespace core
} // namespace dnn
