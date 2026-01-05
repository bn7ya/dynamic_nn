#pragma once

#include <exception>
#include <string>
#include <sstream>

namespace dnn {
namespace exceptions {

/**
 * Base exception class for all DNN exceptions.
 */
class DNNException : public std::exception {
public:
    explicit DNNException(const std::string& message)
        : message_(message) {}

    DNNException(const std::string& message, const std::string& context)
        : message_(message + " [Context: " + context + "]") {}

    const char* what() const noexcept override {
        return message_.c_str();
    }

    const std::string& message() const noexcept {
        return message_;
    }

protected:
    std::string message_;
};

/**
 * Exception for resource-related errors (memory, CPU, disk).
 */
class ResourceException : public DNNException {
public:
    enum class ResourceType {
        Memory,
        CPU,
        GPU,
        DiskSpace
    };

    ResourceException(ResourceType type, size_t required, size_t available,
                      const std::string& context = "")
        : DNNException(build_message(type, required, available), context)
        , type_(type)
        , required_(required)
        , available_(available) {}

    ResourceType resource_type() const noexcept { return type_; }
    size_t required() const noexcept { return required_; }
    size_t available() const noexcept { return available_; }

private:
    static std::string build_message(ResourceType type, size_t required, size_t available) {
        std::ostringstream oss;
        oss << "Resource exhausted: ";
        switch (type) {
            case ResourceType::Memory:
                oss << "Memory";
                break;
            case ResourceType::CPU:
                oss << "CPU";
                break;
            case ResourceType::GPU:
                oss << "GPU";
                break;
            case ResourceType::DiskSpace:
                oss << "Disk Space";
                break;
        }
        oss << " - Required: " << required << ", Available: " << available;
        return oss.str();
    }

    ResourceType type_;
    size_t required_;
    size_t available_;
};

/**
 * Memory allocation exception.
 */
class MemoryException : public ResourceException {
public:
    MemoryException(size_t required, size_t available,
                    const std::string& context = "")
        : ResourceException(ResourceType::Memory, required, available, context) {}
};

/**
 * Training-related exceptions.
 */
class TrainingException : public DNNException {
public:
    enum class Cause {
        NaNDetected,
        InfDetected,
        GradientExplosion,
        GradientVanishing,
        HealthCritical,
        ResourceExhausted,
        InvalidConfiguration,
        ConvergenceFailure
    };

    TrainingException(Cause cause, const std::string& context = "")
        : DNNException(build_message(cause), context)
        , cause_(cause) {}

    Cause cause() const noexcept { return cause_; }

private:
    static std::string build_message(Cause cause) {
        switch (cause) {
            case Cause::NaNDetected:
                return "NaN values detected in computation";
            case Cause::InfDetected:
                return "Infinite values detected in computation";
            case Cause::GradientExplosion:
                return "Gradient explosion detected";
            case Cause::GradientVanishing:
                return "Gradient vanishing detected";
            case Cause::HealthCritical:
                return "Network health is critical";
            case Cause::ResourceExhausted:
                return "Training resources exhausted";
            case Cause::InvalidConfiguration:
                return "Invalid training configuration";
            case Cause::ConvergenceFailure:
                return "Training failed to converge";
            default:
                return "Unknown training error";
        }
    }

    Cause cause_;
};

/**
 * Shape mismatch exception for tensor operations.
 */
class ShapeException : public DNNException {
public:
    ShapeException(const std::string& operation, const std::string& details)
        : DNNException("Shape mismatch in " + operation + ": " + details) {}
};

/**
 * Invalid argument exception.
 */
class InvalidArgumentException : public DNNException {
public:
    InvalidArgumentException(const std::string& argument, const std::string& reason)
        : DNNException("Invalid argument '" + argument + "': " + reason) {}
};

/**
 * IO exception for file operations.
 */
class IOException : public DNNException {
public:
    IOException(const std::string& operation, const std::string& path,
                const std::string& reason = "")
        : DNNException("IO error during " + operation + " on '" + path + "'" +
                      (reason.empty() ? "" : ": " + reason)) {}
};

} // namespace exceptions
} // namespace dnn
