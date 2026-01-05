#pragma once

#include <string>
#include <memory>

namespace dnn {
namespace cqrs {

/**
 * Base class for all commands.
 * Commands represent actions that modify state.
 */
class Command {
public:
    virtual ~Command() = default;

    /**
     * Get command name.
     */
    virtual std::string name() const = 0;

    /**
     * Validate the command.
     */
    virtual bool validate() const = 0;
};

/**
 * Command with a result type.
 */
template<typename TResult>
class CommandWithResult : public Command {
public:
    using ResultType = TResult;
};

} // namespace cqrs
} // namespace dnn
