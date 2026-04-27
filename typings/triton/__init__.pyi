from collections.abc import Callable
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])

__version__: str


class _Kernel:
    def __getitem__(self, grid: Any) -> Callable[..., Any]: ...


def jit(fn: _F | None = ..., **kwargs: Any) -> _Kernel | Callable[[_F], _Kernel]: ...


def cdiv(x: int, y: int) -> int: ...
