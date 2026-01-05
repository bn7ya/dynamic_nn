#pragma once

#include "query.hpp"
#include <unordered_map>
#include <typeindex>
#include <memory>
#include <mutex>

namespace dnn {
namespace cqrs {

/**
 * Query handler base class.
 */
class QueryHandler {
public:
    virtual ~QueryHandler() = default;
};

/**
 * Typed query handler.
 */
template<typename TQuery>
class TypedQueryHandler : public QueryHandler {
public:
    using ResultType = typename TQuery::ResultType;

    virtual ResultType handle(const TQuery& query) const = 0;
};

/**
 * Query bus for dispatching queries to handlers.
 */
class QueryBus {
public:
    /**
     * Register a handler for a query type.
     */
    template<typename TQuery, typename THandler>
    void register_handler(std::shared_ptr<THandler> handler) {
        static_assert(std::is_base_of_v<TypedQueryHandler<TQuery>, THandler>,
                      "Handler must derive from TypedQueryHandler<TQuery>");

        std::lock_guard<std::mutex> lock(mutex_);
        handlers_[std::type_index(typeid(TQuery))] = handler;
    }

    /**
     * Execute a query.
     */
    template<typename TQuery>
    typename TQuery::ResultType query(const TQuery& q) const {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = handlers_.find(std::type_index(typeid(TQuery)));
        if (it == handlers_.end()) {
            throw std::runtime_error("No handler registered for query: " + q.name());
        }

        auto* handler = dynamic_cast<TypedQueryHandler<TQuery>*>(it->second.get());
        if (!handler) {
            throw std::runtime_error("Handler type mismatch for query: " + q.name());
        }

        return handler->handle(q);
    }

    /**
     * Check if a handler is registered for a query type.
     */
    template<typename TQuery>
    bool has_handler() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return handlers_.find(std::type_index(typeid(TQuery))) != handlers_.end();
    }

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::type_index, std::shared_ptr<QueryHandler>> handlers_;
};

} // namespace cqrs
} // namespace dnn
