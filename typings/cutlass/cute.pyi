from collections.abc import Callable, Sequence
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


class Tensor:
    shape: Any
    element_type: Any

    def __getitem__(self, key: Any) -> Any: ...


class Layout:
    pass


class Uint32(int):
    pass


class Int32(int):
    pass


class Float32(float):
    pass


class _Launchable:
    def launch(self, *, grid: Sequence[int], block: Sequence[int]) -> None: ...


class _Kernel:
    def __call__(self, *args: Any, **kwargs: Any) -> _Launchable: ...


class _Arch:
    def thread_idx(self) -> tuple[int, int, int]: ...
    def block_idx(self) -> tuple[int, int, int]: ...


class _Nvgpu:
    class CopyUniversalOp:
        pass


arch: _Arch
nvgpu: _Nvgpu


def jit(fn: _F | None = ..., **kwargs: Any) -> _F | Callable[[_F], _F]: ...


def kernel(fn: _F | None = ..., **kwargs: Any) -> _Kernel | Callable[[_F], _Kernel]: ...


def make_copy_atom(op: Any, dtype: Any) -> Any: ...


def make_tiled_copy_tv(copy_atom: Any, thr_layout: Layout, val_layout: Layout) -> Any: ...


def make_fragment_like(tensor: Any) -> Any: ...


def make_rmem_tensor(shape: Any, dtype: Any) -> Any: ...


def copy(copy_atom: Any, src: Any, dst: Any, *, pred: Any = ...) -> None: ...


def size(value: Any, *, mode: Any = ...) -> int: ...


def make_ordered_layout(shape: tuple[int, ...], *, order: tuple[int, ...]) -> Layout: ...


def make_layout_tv(thr_layout: Layout, val_layout: Layout) -> tuple[Any, Any]: ...


def zipped_divide(tensor: Tensor, tiler: Any) -> Tensor: ...


def make_identity_tensor(shape: Any) -> Tensor: ...


def elem_less(lhs: Any, rhs: Any) -> bool: ...
