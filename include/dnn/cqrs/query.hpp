#pragma once

#include <string>

namespace dnn {
namespace cqrs {

/**
 * Base class for queries.
 * Queries retrieve information without modifying state.
 */
template<typename TResult>
class Query {
public:
    using ResultType = TResult;

    virtual ~Query() = default;

    /**
     * Get query name.
     */
    virtual std::string name() const = 0;
};

} // namespace cqrs
} // namespace dnn
