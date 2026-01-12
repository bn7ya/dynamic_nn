#pragma once

#include "../core/tensor.hpp"
#include <memory>
#include <string>
#include <map>

namespace dnn {
namespace io {

using core::Tensor;

/**
 * Input types supported by the network.
 */
enum class InputType {
    Embedding,    // Pre-computed embeddings
    Vector,       // 1D numeric arrays
    Array,        // N-D numeric arrays
    Image,        // Image files
    Video,        // Video files
    Audio,        // Audio files
    RawText       // Raw text strings
};

/**
 * Base input adapter class.
 */
template<typename T = float>
class InputAdapter {
public:
    virtual ~InputAdapter() = default;

    /**
     * Get input type.
     */
    virtual InputType type() const = 0;

    /**
     * Adapt raw data to tensor.
     */
    virtual Tensor<T> adapt(const void* raw_data, size_t size) const = 0;

    /**
     * Get output shape.
     */
    virtual std::vector<size_t> output_shape() const = 0;

    /**
     * Factory method.
     */
    static std::unique_ptr<InputAdapter> create(
        InputType type,
        const std::map<std::string, std::string>& config = {});
};

/**
 * Vector/Array adapter - for numeric data.
 */
template<typename T = float>
class VectorAdapter : public InputAdapter<T> {
public:
    explicit VectorAdapter(const std::vector<size_t>& shape)
        : shape_(shape) {}

    InputType type() const override { return InputType::Vector; }

    Tensor<T> adapt(const void* raw_data, size_t size) const override {
        size_t expected_size = 1;
        for (size_t dim : shape_) {
            expected_size *= dim;
        }

        if (size < expected_size * sizeof(T)) {
            throw std::invalid_argument("Input data too small");
        }

        Tensor<T> result(shape_);
        const T* data = static_cast<const T*>(raw_data);
        for (size_t i = 0; i < expected_size; ++i) {
            result[i] = data[i];
        }
        return result;
    }

    std::vector<size_t> output_shape() const override {
        return shape_;
    }

private:
    std::vector<size_t> shape_;
};

/**
 * Image adapter configuration.
 */
struct ImageConfig {
    size_t target_width = 224;
    size_t target_height = 224;
    size_t channels = 3;
    bool normalize = true;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
};

/**
 * Image adapter - for image data.
 * Note: Full implementation would need image loading library.
 */
template<typename T = float>
class ImageAdapter : public InputAdapter<T> {
public:
    explicit ImageAdapter(const ImageConfig& config = ImageConfig())
        : config_(config) {}

    InputType type() const override { return InputType::Image; }

    Tensor<T> adapt(const void* raw_data, size_t size) const override {
        // This is a placeholder implementation
        // Full implementation would decode image and resize

        // Assume raw_data is already resized RGB pixels
        size_t expected = config_.channels * config_.target_height * config_.target_width;

        Tensor<T> result(std::vector<size_t>{config_.channels, config_.target_height, config_.target_width});

        const uint8_t* pixels = static_cast<const uint8_t*>(raw_data);
        size_t idx = 0;

        for (size_t c = 0; c < config_.channels; ++c) {
            for (size_t h = 0; h < config_.target_height; ++h) {
                for (size_t w = 0; w < config_.target_width; ++w) {
                    T value = static_cast<T>(pixels[idx++]) / T(255);
                    if (config_.normalize) {
                        value = (value - config_.mean[c]) / config_.std[c];
                    }
                    result.at({c, h, w}) = value;
                }
            }
        }

        return result;
    }

    std::vector<size_t> output_shape() const override {
        return {config_.channels, config_.target_height, config_.target_width};
    }

private:
    ImageConfig config_;
};

/**
 * Text adapter configuration.
 */
struct TextConfig {
    size_t max_length = 512;
    size_t vocab_size = 30000;
    bool lowercase = true;
};

/**
 * Text adapter - for raw text.
 * Note: Full implementation would need tokenizer.
 */
template<typename T = float>
class TextAdapter : public InputAdapter<T> {
public:
    explicit TextAdapter(const TextConfig& config = TextConfig())
        : config_(config) {}

    InputType type() const override { return InputType::RawText; }

    Tensor<T> adapt(const void* raw_data, size_t size) const override {
        // This is a simplified placeholder
        // Full implementation would tokenize and encode text

        const char* text = static_cast<const char*>(raw_data);
        std::string str(text, size);

        // Simple character-level encoding (placeholder)
        Tensor<T> result(std::vector<size_t>{config_.max_length});
        result.fill(T(0));  // Padding

        for (size_t i = 0; i < std::min(size, config_.max_length); ++i) {
            unsigned char c = str[i];
            if (config_.lowercase && c >= 'A' && c <= 'Z') {
                c = c - 'A' + 'a';
            }
            result[i] = static_cast<T>(c) / T(256);  // Normalize
        }

        return result;
    }

    std::vector<size_t> output_shape() const override {
        return {config_.max_length};
    }

private:
    TextConfig config_;
};

// Factory implementation
template<typename T>
std::unique_ptr<InputAdapter<T>> InputAdapter<T>::create(
        InputType type,
        const std::map<std::string, std::string>& config) {

    switch (type) {
        case InputType::Vector:
        case InputType::Array: {
            std::vector<size_t> shape = {1};  // Default
            if (config.count("shape")) {
                // Parse shape from config
            }
            return std::make_unique<VectorAdapter<T>>(shape);
        }

        case InputType::Image: {
            ImageConfig img_config;
            if (config.count("width")) {
                img_config.target_width = std::stoull(config.at("width"));
            }
            if (config.count("height")) {
                img_config.target_height = std::stoull(config.at("height"));
            }
            return std::make_unique<ImageAdapter<T>>(img_config);
        }

        case InputType::RawText: {
            TextConfig text_config;
            if (config.count("max_length")) {
                text_config.max_length = std::stoull(config.at("max_length"));
            }
            return std::make_unique<TextAdapter<T>>(text_config);
        }

        default:
            return std::make_unique<VectorAdapter<T>>(std::vector<size_t>{1});
    }
}

} // namespace io
} // namespace dnn
