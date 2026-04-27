from typing import Any


class constexpr:
    pass


float16: Any
float32: Any
int32: Any
uint32: Any


def program_id(axis: int) -> int: ...


def arange(start: int, end: int) -> Any: ...


def load(pointer: Any, mask: Any = ..., other: Any = ...) -> Any: ...


def store(pointer: Any, value: Any, mask: Any = ...) -> None: ...
