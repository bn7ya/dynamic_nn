/**
 * pybind11 bindings for the transformer C++/CUDA ops.
 *
 * Exposed as a submodule ``_dnn_core.transformer_ops``. All ops are
 * NumPy in / NumPy out and release the GIL around the actual work.
 *
 * Backend selection: each op has a ``_cpu`` and (when CUDA is built)
 * a ``_cuda`` variant. The dispatcher in
 * ``python/pydnn/transformer/_backend.py`` picks one. CPU is always
 * available; CUDA availability is reported by ``cuda_available()``.
 *
 * Numerics: every op matches the NumPy reference in
 * ``python/pydnn/transformer/autograd.py``.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "dnn/transformer/ops.hpp"

#ifdef DNN_ENABLE_CUDA
#include "dnn/cuda/cuda_common.hpp"
#include <cuda_runtime.h>
#endif

namespace py = pybind11;
namespace tx = dnn::transformer;

namespace {

py::array_t<float> ensure_f32(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    return arr;
}

py::array_t<int64_t> ensure_i64(py::array_t<int64_t, py::array::c_style | py::array::forcecast> arr) {
    return arr;
}

inline const float* fptr(const py::array_t<float>& a) {
    return static_cast<const float*>(a.request().ptr);
}
inline float* fptr_mut(py::array_t<float>& a) {
    return static_cast<float*>(a.request().ptr);
}
inline const int64_t* iptr(const py::array_t<int64_t>& a) {
    return static_cast<const int64_t*>(a.request().ptr);
}

py::array_t<float> alloc_f32(std::vector<py::ssize_t> shape) {
    return py::array_t<float>(shape);
}

// Compute leading-dim broadcast shape + per-axis batch strides (in
// elements) for a Python-style numpy matmul. ``a_shape`` and
// ``b_shape`` already exclude the trailing two matrix dims.
struct BroadcastInfo {
    std::vector<int64_t> dims;
    std::vector<int64_t> a_strides;
    std::vector<int64_t> b_strides;
};

BroadcastInfo build_broadcast(const std::vector<int64_t>& a_lead,
                              const std::vector<int64_t>& b_lead,
                              int64_t a_mat_size, int64_t b_mat_size) {
    int rank = int(std::max(a_lead.size(), b_lead.size()));
    std::vector<int64_t> a_pad(rank, 1), b_pad(rank, 1);
    for (int i = 0; i < int(a_lead.size()); ++i) {
        a_pad[rank - int(a_lead.size()) + i] = a_lead[i];
    }
    for (int i = 0; i < int(b_lead.size()); ++i) {
        b_pad[rank - int(b_lead.size()) + i] = b_lead[i];
    }
    BroadcastInfo info;
    info.dims.resize(rank);
    info.a_strides.resize(rank);
    info.b_strides.resize(rank);
    int64_t a_acc = a_mat_size, b_acc = b_mat_size;
    for (int axis = rank - 1; axis >= 0; --axis) {
        int64_t da = a_pad[axis], db = b_pad[axis];
        if (da != db && da != 1 && db != 1) {
            throw std::invalid_argument(
                "matmul leading dims are not broadcast-compatible");
        }
        info.dims[axis] = std::max(da, db);
        info.a_strides[axis] = (da == 1) ? 0 : a_acc;
        info.b_strides[axis] = (db == 1) ? 0 : b_acc;
        a_acc *= da;
        b_acc *= db;
    }
    return info;
}

#ifdef DNN_ENABLE_CUDA
// RAII device buffer scoped to one op.
struct DeviceBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    DeviceBuf() = default;
    explicit DeviceBuf(size_t n_bytes) : bytes(n_bytes) {
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
        o.ptr = nullptr; o.bytes = 0;
    }
    ~DeviceBuf() { if (ptr) cudaFree(ptr); }
};

template <typename T>
DeviceBuf upload(const py::array_t<T>& host) {
    auto buf = host.request();
    DeviceBuf d(buf.size * sizeof(T));
    cudaMemcpy(d.ptr, buf.ptr, d.bytes, cudaMemcpyHostToDevice);
    return d;
}

template <typename T>
void download(const DeviceBuf& d, py::array_t<T>& host) {
    auto buf = host.request();
    cudaMemcpy(buf.ptr, d.ptr, d.bytes, cudaMemcpyDeviceToHost);
}
#endif

}  // namespace

// ============================================================
// CPU ops (numpy in / numpy out)
// ============================================================

static py::array_t<float> matmul_cpu(py::array_t<float> a_in, py::array_t<float> b_in,
                                     bool transpose_a, bool transpose_b) {
    auto a = ensure_f32(a_in);
    auto b = ensure_f32(b_in);
    auto a_buf = a.request();
    auto b_buf = b.request();
    if (a_buf.ndim < 2 || b_buf.ndim < 2) {
        throw std::invalid_argument("matmul: both inputs must be at least 2-D");
    }
    int64_t M = a_buf.shape[a_buf.ndim - 2];
    int64_t K = a_buf.shape[a_buf.ndim - 1];
    int64_t Kb = b_buf.shape[b_buf.ndim - 2];
    int64_t N = b_buf.shape[b_buf.ndim - 1];
    if (transpose_a) std::swap(M, K);
    if (transpose_b) std::swap(Kb, N);
    if (K != Kb) {
        throw std::invalid_argument("matmul: inner dims do not match");
    }

    std::vector<int64_t> a_lead(a_buf.shape.begin(), a_buf.shape.end() - 2);
    std::vector<int64_t> b_lead(b_buf.shape.begin(), b_buf.shape.end() - 2);
    auto info = build_broadcast(
        a_lead, b_lead,
        a_buf.shape[a_buf.ndim - 2] * a_buf.shape[a_buf.ndim - 1],
        b_buf.shape[b_buf.ndim - 2] * b_buf.shape[b_buf.ndim - 1]);

    std::vector<py::ssize_t> out_shape(info.dims.begin(), info.dims.end());
    out_shape.push_back(py::ssize_t(M));
    out_shape.push_back(py::ssize_t(N));
    auto out = alloc_f32(out_shape);

    {
        py::gil_scoped_release rel;
        tx::matmul_batched_cpu(
            fptr(a), fptr(b), fptr_mut(out),
            info.dims.data(), int(info.dims.size()),
            info.a_strides.data(), info.b_strides.data(),
            M, K, N, transpose_a, transpose_b);
    }
    return out;
}

static py::array_t<float> softmax_cpu(py::array_t<float> x_in) {
    auto x = ensure_f32(x_in);
    auto buf = x.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    auto out = py::array_t<float>(buf.shape);
    {
        py::gil_scoped_release rel;
        tx::softmax_lastdim_cpu(fptr(x), fptr_mut(out), rows, cols);
    }
    return out;
}

static py::array_t<float> softmax_backward_cpu(py::array_t<float> y_in,
                                               py::array_t<float> dy_in) {
    auto y = ensure_f32(y_in);
    auto dy = ensure_f32(dy_in);
    auto buf = y.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    auto dx = py::array_t<float>(buf.shape);
    {
        py::gil_scoped_release rel;
        tx::softmax_lastdim_backward_cpu(fptr(y), fptr(dy), fptr_mut(dx), rows, cols);
    }
    return dx;
}

static py::tuple layernorm_forward_cpu(py::array_t<float> x_in,
                                       py::array_t<float> gamma_in,
                                       py::array_t<float> beta_in,
                                       float eps) {
    auto x = ensure_f32(x_in);
    auto g = ensure_f32(gamma_in);
    auto b = ensure_f32(beta_in);
    auto buf = x.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    auto y = py::array_t<float>(buf.shape);
    auto mean = py::array_t<float>({py::ssize_t(rows)});
    auto inv_std = py::array_t<float>({py::ssize_t(rows)});
    {
        py::gil_scoped_release rel;
        tx::layernorm_forward_cpu(
            fptr(x), fptr(g), fptr(b),
            fptr_mut(y), fptr_mut(mean), fptr_mut(inv_std),
            rows, cols, eps);
    }
    return py::make_tuple(y, mean, inv_std);
}

static py::tuple layernorm_backward_cpu(py::array_t<float> x_in,
                                        py::array_t<float> gamma_in,
                                        py::array_t<float> mean_in,
                                        py::array_t<float> inv_std_in,
                                        py::array_t<float> dy_in) {
    auto x = ensure_f32(x_in);
    auto g = ensure_f32(gamma_in);
    auto m = ensure_f32(mean_in);
    auto isd = ensure_f32(inv_std_in);
    auto dy = ensure_f32(dy_in);
    auto buf = x.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    auto dx = py::array_t<float>(buf.shape);
    auto dgamma = py::array_t<float>({py::ssize_t(cols)});
    auto dbeta = py::array_t<float>({py::ssize_t(cols)});
    {
        py::gil_scoped_release rel;
        tx::layernorm_backward_cpu(
            fptr(x), fptr(g), fptr(m), fptr(isd), fptr(dy),
            fptr_mut(dx), fptr_mut(dgamma), fptr_mut(dbeta),
            rows, cols);
    }
    return py::make_tuple(dx, dgamma, dbeta);
}

static py::tuple rmsnorm_forward_cpu(py::array_t<float> x_in,
                                     py::array_t<float> gamma_in,
                                     float eps) {
    auto x = ensure_f32(x_in);
    auto g = ensure_f32(gamma_in);
    auto buf = x.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    auto y = py::array_t<float>(buf.shape);
    auto inv_rms = py::array_t<float>({py::ssize_t(rows)});
    {
        py::gil_scoped_release rel;
        tx::rmsnorm_forward_cpu(
            fptr(x), fptr(g), fptr_mut(y), fptr_mut(inv_rms),
            rows, cols, eps);
    }
    return py::make_tuple(y, inv_rms);
}

static py::tuple rmsnorm_backward_cpu(py::array_t<float> x_in,
                                      py::array_t<float> gamma_in,
                                      py::array_t<float> inv_rms_in,
                                      py::array_t<float> dy_in) {
    auto x = ensure_f32(x_in);
    auto g = ensure_f32(gamma_in);
    auto inv = ensure_f32(inv_rms_in);
    auto dy = ensure_f32(dy_in);
    auto buf = x.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    auto dx = py::array_t<float>(buf.shape);
    auto dgamma = py::array_t<float>({py::ssize_t(cols)});
    {
        py::gil_scoped_release rel;
        tx::rmsnorm_backward_cpu(
            fptr(x), fptr(g), fptr(inv), fptr(dy),
            fptr_mut(dx), fptr_mut(dgamma),
            rows, cols);
    }
    return py::make_tuple(dx, dgamma);
}

#define DEF_ELEMENT_ACT(NAME, FWD_FN, BWD_FN)                                     \
    static py::array_t<float> NAME##_cpu(py::array_t<float> x_in) {               \
        auto x = ensure_f32(x_in);                                                \
        auto buf = x.request();                                                   \
        auto out = py::array_t<float>(buf.shape);                                 \
        {                                                                         \
            py::gil_scoped_release rel;                                           \
            FWD_FN(fptr(x), fptr_mut(out), int64_t(buf.size));                    \
        }                                                                         \
        return out;                                                               \
    }                                                                             \
    static py::array_t<float> NAME##_backward_cpu(py::array_t<float> x_in,        \
                                                  py::array_t<float> dy_in) {     \
        auto x = ensure_f32(x_in);                                                \
        auto dy = ensure_f32(dy_in);                                              \
        auto buf = x.request();                                                   \
        auto dx = py::array_t<float>(buf.shape);                                  \
        {                                                                         \
            py::gil_scoped_release rel;                                           \
            BWD_FN(fptr(x), fptr(dy), fptr_mut(dx), int64_t(buf.size));           \
        }                                                                         \
        return dx;                                                                \
    }

DEF_ELEMENT_ACT(gelu, tx::gelu_forward_cpu, tx::gelu_backward_cpu)
DEF_ELEMENT_ACT(silu, tx::silu_forward_cpu, tx::silu_backward_cpu)

static py::array_t<float> embedding_forward_cpu(py::array_t<float> weight_in,
                                                py::array_t<int64_t> ids_in) {
    auto w = ensure_f32(weight_in);
    auto ids = ensure_i64(ids_in);
    auto wbuf = w.request();
    auto ibuf = ids.request();
    int64_t vocab = wbuf.shape[0];
    int64_t dim = wbuf.shape[1];
    int64_t n = ibuf.size;
    std::vector<py::ssize_t> out_shape(ibuf.shape.begin(), ibuf.shape.end());
    out_shape.push_back(py::ssize_t(dim));
    auto out = alloc_f32(out_shape);
    {
        py::gil_scoped_release rel;
        tx::embedding_forward_cpu(fptr(w), iptr(ids), fptr_mut(out), n, dim, vocab);
    }
    return out;
}

static py::array_t<float> embedding_backward_cpu(py::array_t<float> dy_in,
                                                 py::array_t<int64_t> ids_in,
                                                 int64_t vocab, int64_t dim) {
    auto dy = ensure_f32(dy_in);
    auto ids = ensure_i64(ids_in);
    auto dweight = py::array_t<float>({py::ssize_t(vocab), py::ssize_t(dim)});
    {
        auto buf = dweight.request();
        std::memset(buf.ptr, 0, sizeof(float) * buf.size);
    }
    int64_t n = ids.request().size;
    {
        py::gil_scoped_release rel;
        tx::embedding_backward_cpu(fptr(dy), iptr(ids), fptr_mut(dweight),
                                   n, dim, vocab);
    }
    return dweight;
}

static py::tuple xent_forward_cpu(py::array_t<float> logits_in,
                                  py::array_t<int64_t> targets_in,
                                  int64_t ignore_index) {
    auto logits = ensure_f32(logits_in);
    auto targets = ensure_i64(targets_in);
    auto lbuf = logits.request();
    if (lbuf.ndim < 1) {
        throw std::invalid_argument("xent: logits must be at least 1-D");
    }
    int64_t v = lbuf.shape[lbuf.ndim - 1];
    int64_t n = 1;
    for (int i = 0; i < lbuf.ndim - 1; ++i) n *= lbuf.shape[i];
    auto log_probs = py::array_t<float>(lbuf.shape);
    int64_t valid = 0;
    float loss = 0.0f;
    {
        py::gil_scoped_release rel;
        loss = tx::softmax_xent_forward_cpu(
            fptr(logits), iptr(targets),
            fptr_mut(log_probs), &valid, n, v, ignore_index);
    }
    return py::make_tuple(loss, log_probs, valid);
}

static py::array_t<float> xent_backward_cpu(py::array_t<float> log_probs_in,
                                            py::array_t<int64_t> targets_in,
                                            float grad_loss,
                                            int64_t valid_count,
                                            int64_t ignore_index) {
    auto lp = ensure_f32(log_probs_in);
    auto targets = ensure_i64(targets_in);
    auto buf = lp.request();
    int64_t v = buf.shape[buf.ndim - 1];
    int64_t n = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) n *= buf.shape[i];
    auto dlogits = py::array_t<float>(buf.shape);
    {
        py::gil_scoped_release rel;
        tx::softmax_xent_backward_cpu(fptr(lp), iptr(targets),
                                      grad_loss, valid_count,
                                      fptr_mut(dlogits),
                                      n, v, ignore_index);
    }
    return dlogits;
}

// ============================================================
// CUDA wrappers (host numpy in/out, copy via DeviceBuf)
// ============================================================

#ifdef DNN_ENABLE_CUDA

static py::array_t<float> matmul_cuda(py::array_t<float> a_in,
                                      py::array_t<float> b_in,
                                      bool transpose_a, bool transpose_b) {
    auto a = ensure_f32(a_in);
    auto b = ensure_f32(b_in);
    auto a_buf = a.request();
    auto b_buf = b.request();
    int64_t M = a_buf.shape[a_buf.ndim - 2];
    int64_t K = a_buf.shape[a_buf.ndim - 1];
    int64_t Kb = b_buf.shape[b_buf.ndim - 2];
    int64_t N = b_buf.shape[b_buf.ndim - 1];
    if (transpose_a) std::swap(M, K);
    if (transpose_b) std::swap(Kb, N);
    std::vector<int64_t> a_lead(a_buf.shape.begin(), a_buf.shape.end() - 2);
    std::vector<int64_t> b_lead(b_buf.shape.begin(), b_buf.shape.end() - 2);
    auto info = build_broadcast(
        a_lead, b_lead,
        a_buf.shape[a_buf.ndim - 2] * a_buf.shape[a_buf.ndim - 1],
        b_buf.shape[b_buf.ndim - 2] * b_buf.shape[b_buf.ndim - 1]);
    std::vector<py::ssize_t> out_shape(info.dims.begin(), info.dims.end());
    out_shape.push_back(py::ssize_t(M));
    out_shape.push_back(py::ssize_t(N));
    auto out = alloc_f32(out_shape);

    DeviceBuf da = upload(a);
    DeviceBuf db = upload(b);
    int64_t batch = 1;
    for (auto d : info.dims) batch *= d;
    DeviceBuf dc(batch * M * N * sizeof(float));
    {
        py::gil_scoped_release rel;
        tx::matmul_batched_cuda(
            (const float*)da.ptr, (const float*)db.ptr, (float*)dc.ptr,
            info.dims.data(), int(info.dims.size()),
            info.a_strides.data(), info.b_strides.data(),
            M, K, N, transpose_a, transpose_b);
        cudaDeviceSynchronize();
    }
    download(dc, out);
    return out;
}

#define ROUND_TRIP_F(NAME, FN)                                                    \
    static py::array_t<float> NAME(py::array_t<float> x_in) {                     \
        auto x = ensure_f32(x_in);                                                \
        auto buf = x.request();                                                   \
        DeviceBuf dx_in = upload(x);                                              \
        DeviceBuf dy(buf.size * sizeof(float));                                   \
        int64_t cols = buf.shape.back();                                          \
        int64_t rows = 1;                                                         \
        for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];              \
        {                                                                         \
            py::gil_scoped_release rel;                                           \
            FN((const float*)dx_in.ptr, (float*)dy.ptr, rows, cols);              \
            cudaDeviceSynchronize();                                              \
        }                                                                         \
        auto out = py::array_t<float>(buf.shape);                                 \
        download(dy, out);                                                        \
        return out;                                                               \
    }

ROUND_TRIP_F(softmax_cuda, tx::softmax_lastdim_cuda)

static py::array_t<float> softmax_backward_cuda(py::array_t<float> y_in,
                                                py::array_t<float> dy_in) {
    auto y = ensure_f32(y_in);
    auto dy = ensure_f32(dy_in);
    auto buf = y.request();
    int64_t cols = buf.shape.back();
    int64_t rows = 1;
    for (int i = 0; i < buf.ndim - 1; ++i) rows *= buf.shape[i];
    DeviceBuf d_y = upload(y), d_dy = upload(dy);
    DeviceBuf d_dx(buf.size * sizeof(float));
    {
        py::gil_scoped_release rel;
        tx::softmax_lastdim_backward_cuda(
            (const float*)d_y.ptr, (const float*)d_dy.ptr,
            (float*)d_dx.ptr, rows, cols);
        cudaDeviceSynchronize();
    }
    auto dx = py::array_t<float>(buf.shape);
    download(d_dx, dx);
    return dx;
}

#define DEF_ACT_CUDA(NAME, FWD_FN, BWD_FN)                                        \
    static py::array_t<float> NAME##_cuda(py::array_t<float> x_in) {              \
        auto x = ensure_f32(x_in);                                                \
        auto buf = x.request();                                                   \
        DeviceBuf d_x = upload(x);                                                \
        DeviceBuf d_y(buf.size * sizeof(float));                                  \
        {                                                                         \
            py::gil_scoped_release rel;                                           \
            FWD_FN((const float*)d_x.ptr, (float*)d_y.ptr, int64_t(buf.size));    \
            cudaDeviceSynchronize();                                              \
        }                                                                         \
        auto out = py::array_t<float>(buf.shape);                                 \
        download(d_y, out);                                                       \
        return out;                                                               \
    }                                                                             \
    static py::array_t<float> NAME##_backward_cuda(py::array_t<float> x_in,       \
                                                   py::array_t<float> dy_in) {    \
        auto x = ensure_f32(x_in);                                                \
        auto dy = ensure_f32(dy_in);                                              \
        auto buf = x.request();                                                   \
        DeviceBuf d_x = upload(x), d_dy = upload(dy);                             \
        DeviceBuf d_dx(buf.size * sizeof(float));                                 \
        {                                                                         \
            py::gil_scoped_release rel;                                           \
            BWD_FN((const float*)d_x.ptr, (const float*)d_dy.ptr,                 \
                   (float*)d_dx.ptr, int64_t(buf.size));                          \
            cudaDeviceSynchronize();                                              \
        }                                                                         \
        auto out = py::array_t<float>(buf.shape);                                 \
        download(d_dx, out);                                                      \
        return out;                                                               \
    }

DEF_ACT_CUDA(gelu, tx::gelu_forward_cuda, tx::gelu_backward_cuda)
DEF_ACT_CUDA(silu, tx::silu_forward_cuda, tx::silu_backward_cuda)

#endif  // DNN_ENABLE_CUDA

// ============================================================
// Module registration
// ============================================================

void register_transformer_ops(py::module_& m) {
    auto sub = m.def_submodule("transformer_ops",
        "Native transformer ops powering pydnn.transformer's optional backend.");

    sub.attr("CUDA_AVAILABLE") = bool(tx::cuda_transformer_available());

    // CPU
    sub.def("matmul_cpu", &matmul_cpu,
            py::arg("a"), py::arg("b"),
            py::arg("transpose_a") = false, py::arg("transpose_b") = false);
    sub.def("softmax_cpu", &softmax_cpu, py::arg("x"));
    sub.def("softmax_backward_cpu", &softmax_backward_cpu,
            py::arg("y"), py::arg("dy"));
    sub.def("layernorm_forward_cpu", &layernorm_forward_cpu,
            py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f);
    sub.def("layernorm_backward_cpu", &layernorm_backward_cpu,
            py::arg("x"), py::arg("gamma"), py::arg("mean"),
            py::arg("inv_std"), py::arg("dy"));
    sub.def("rmsnorm_forward_cpu", &rmsnorm_forward_cpu,
            py::arg("x"), py::arg("gamma"), py::arg("eps") = 1e-6f);
    sub.def("rmsnorm_backward_cpu", &rmsnorm_backward_cpu,
            py::arg("x"), py::arg("gamma"), py::arg("inv_rms"), py::arg("dy"));
    sub.def("gelu_cpu", &gelu_cpu, py::arg("x"));
    sub.def("gelu_backward_cpu", &gelu_backward_cpu, py::arg("x"), py::arg("dy"));
    sub.def("silu_cpu", &silu_cpu, py::arg("x"));
    sub.def("silu_backward_cpu", &silu_backward_cpu, py::arg("x"), py::arg("dy"));
    sub.def("embedding_forward_cpu", &embedding_forward_cpu,
            py::arg("weight"), py::arg("ids"));
    sub.def("embedding_backward_cpu", &embedding_backward_cpu,
            py::arg("dy"), py::arg("ids"), py::arg("vocab"), py::arg("dim"));
    sub.def("xent_forward_cpu", &xent_forward_cpu,
            py::arg("logits"), py::arg("targets"), py::arg("ignore_index") = -100);
    sub.def("xent_backward_cpu", &xent_backward_cpu,
            py::arg("log_probs"), py::arg("targets"),
            py::arg("grad_loss"), py::arg("valid_count"),
            py::arg("ignore_index") = -100);

#ifdef DNN_ENABLE_CUDA
    sub.def("matmul_cuda", &matmul_cuda,
            py::arg("a"), py::arg("b"),
            py::arg("transpose_a") = false, py::arg("transpose_b") = false);
    sub.def("softmax_cuda", &softmax_cuda, py::arg("x"));
    sub.def("softmax_backward_cuda", &softmax_backward_cuda,
            py::arg("y"), py::arg("dy"));
    sub.def("gelu_cuda", &gelu_cuda, py::arg("x"));
    sub.def("gelu_backward_cuda", &gelu_backward_cuda, py::arg("x"), py::arg("dy"));
    sub.def("silu_cuda", &silu_cuda, py::arg("x"));
    sub.def("silu_backward_cuda", &silu_backward_cuda, py::arg("x"), py::arg("dy"));
#endif
}
