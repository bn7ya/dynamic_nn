#pragma once

#include "command.hpp"
#include <unordered_map>
#include <typeindex>
#include <functional>
#include <memory>
#include <mutex>
#include <future>

namespace dnn {
namespace cqrs {

/**
 * Command handler base class.
 */
class CommandHandler {
public:
    virtual ~CommandHandler() = default;
};

/**
 * Typed command handler.
 */
template<typename TCommand>
class TypedCommandHandler : public CommandHandler {
public:
    using ResultType = typename TCommand::ResultType;

    virtual ResultType handle(const TCommand& command) = 0;
};

/**
 * Command bus for dispatching commands to handlers.
 */
class CommandBus {
public:
    /**
     * Register a handler for a command type.
     */
    template<typename TCommand, typename THandler>
    void register_handler(std::shared_ptr<THandler> handler) {
        static_assert(std::is_base_of_v<TypedCommandHandler<TCommand>, THandler>,
                      "Handler must derive from TypedCommandHandler<TCommand>");

        std::lock_guard<std::mutex> lock(mutex_);
        handlers_[std::type_index(typeid(TCommand))] = handler;
    }

    /**
     * Execute a command synchronously.
     */
    template<typename TCommand>
    typename TCommand::ResultType execute(const TCommand& command) {
        if (!command.validate()) {
            throw std::invalid_argument("Command validation failed: " + command.name());
        }

        std::lock_guard<std::mutex> lock(mutex_);

        auto it = handlers_.find(std::type_index(typeid(TCommand)));
        if (it == handlers_.end()) {
            throw std::runtime_error("No handler registered for command: " + command.name());
        }

        auto* handler = dynamic_cast<TypedCommandHandler<TCommand>*>(it->second.get());
        if (!handler) {
            throw std::runtime_error("Handler type mismatch for command: " + command.name());
        }

        return handler->handle(command);
    }

    /**
     * Execute a command asynchronously.
     */
    template<typename TCommand>
    std::future<typename TCommand::ResultType> execute_async(const TCommand& command) {
        return std::async(std::launch::async, [this, command]() {
            return execute(command);
        });
    }

    /**
     * Check if a handler is registered for a command type.
     */
    template<typename TCommand>
    bool has_handler() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return handlers_.find(std::type_index(typeid(TCommand))) != handlers_.end();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::type_index, std::shared_ptr<CommandHandler>> handlers_;
};

} // namespace cqrs
} // namespace dnn
