"""
Base ``Module`` class — a tiny analogue of ``torch.nn.Module``.

Modules own ``Parameter`` tensors and (optionally) child modules.
``parameters()`` walks the tree so optimizers can iterate over every
learnable tensor; ``train()`` / ``eval()`` flip a ``training`` flag
that ops like Dropout consult.
"""

from __future__ import annotations

from typing import Iterator, List

from .autograd import Parameter, Tensor


class Module:
    """Base class for all transformer building blocks."""

    def __init__(self) -> None:
        # Use object.__setattr__ to avoid triggering our overridden __setattr__
        # for these bookkeeping dicts.
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "training", True)

    # -------------------- attribute capture --------------------

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, (list, tuple)) and value and all(
            isinstance(v, Module) for v in value
        ):
            # Register list-of-modules under indexed names so parameters() finds them.
            self._modules[name] = ModuleList(value)
            object.__setattr__(self, name, self._modules[name])
            return
        object.__setattr__(self, name, value)

    # -------------------- traversal --------------------

    def parameters(self) -> Iterator[Parameter]:
        for p in self._parameters.values():
            yield p
        for m in self._modules.values():
            for p in m.parameters():
                yield p

    def named_parameters(self, prefix: str = "") -> Iterator:
        for name, p in self._parameters.items():
            yield (f"{prefix}.{name}" if prefix else name), p
        for name, m in self._modules.items():
            sub = f"{prefix}.{name}" if prefix else name
            for n, p in m.named_parameters(sub):
                yield n, p

    def modules(self) -> Iterator["Module"]:
        yield self
        for m in self._modules.values():
            for sub in m.modules():
                yield sub

    # -------------------- training mode --------------------

    def train(self, mode: bool = True) -> "Module":
        object.__setattr__(self, "training", mode)
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self) -> "Module":
        return self.train(False)

    # -------------------- gradients --------------------

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        return sum(int(p.data.size) for p in self.parameters())

    # -------------------- call --------------------

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ModuleList(Module):
    """An ordered container of submodules accessible by index."""

    def __init__(self, modules: List[Module] = None) -> None:
        super().__init__()
        self._items: List[Module] = []
        if modules:
            for m in modules:
                self.append(m)

    def append(self, module: Module) -> None:
        idx = len(self._items)
        self._items.append(module)
        self._modules[str(idx)] = module

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx: int) -> Module:
        return self._items[idx]

    def __len__(self) -> int:
        return len(self._items)
