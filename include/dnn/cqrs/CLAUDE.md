# CLAUDE.md — `include/dnn/cqrs/`

## Purpose

Command/Query Responsibility Segregation infrastructure: a `CommandBus`
that dispatches strongly-typed commands to registered handlers, and a
parallel `QueryBus` for read-only requests. **Currently dormant** —
the types compile and link, but no production code path uses the
buses yet. They exist as a backbone for future async dispatch.

## Files

| File | Role |
|---|---|
| `command.hpp` | `Command` base + `TypedCommand<TResult>` template. |
| `command_bus.hpp` | `CommandBus::register_handler<TCmd>` / `execute<TCmd>` / `execute_async<TCmd>`. `[scaffolding]` |
| `query.hpp` | `Query` base + `TypedQuery<TResult>` template. |
| `query_bus.hpp` | Mirror of `CommandBus` for queries. `[scaffolding]` |

`src/cqrs/{command_bus,query_bus}.cpp` provide explicit instantiations.

## Invariants

- **The bus mutex protects the handler map only.** It does not protect
  handler bodies. If a handler touches shared state, the handler is
  responsible for its own synchronisation. (`command_bus.hpp:46-71`.)
- **Commands validate before dispatch.** `CommandBus::execute` calls
  `command.validate()` and throws `std::invalid_argument` on failure.
  Don't bypass it.
- **`execute_async` returns `std::future`.** It uses
  `std::launch::async`, which spawns a fresh thread per call — fine
  for low frequency, not OK if you start dispatching per-batch
  commands. Pool first if that changes.

## Maintenance notes

- This is the only place in the repo that ships a generic async
  dispatch primitive. The `StageController`
  (`include/dnn/training/runtime/`) **does not** currently route
  through `CommandBus`; it calls workers directly. The original plan
  was to define `WakeStageCommand`, `MutateTopologyCommand`, etc. and
  dispatch through here. Promoting the runtime to actually use this
  infrastructure is a real follow-up — when you do, the runtime's
  CLAUDE.md becomes the relevant cross-reference.
- Removing this directory would shrink the build; **don't** without
  first checking that nothing under `include/dnn/training/runtime/`
  has started using it.

## Memory & reliability notes

- Each `register_handler` call stores a `std::shared_ptr<CommandHandler>`
  in the map. Handlers live as long as the bus. Don't hand out raw
  pointers.
- `execute_async`'s `std::async` policy doesn't cap concurrency;
  unbounded callers can overcommit threads. If this becomes a hot
  path, swap in a bounded thread pool.

## Cross-refs

- The runtime that eventually uses this lives in
  [`include/dnn/training/runtime/CLAUDE.md`](../training/runtime/CLAUDE.md).
- General threading rules are in the root
  [`CLAUDE.md`](../../../CLAUDE.md) under "Reliability".

## Updating this file

When the runtime starts dispatching real commands here, drop the
`[scaffolding]` tags and add a "Live commands" section listing the
concrete `Command` types. If this directory is deleted, remove its
row from the root feature index. Keep this file ≤120 lines.
