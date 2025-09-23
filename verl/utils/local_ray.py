"""
Lightweight local shim for Ray-like APIs to enable single-process execution
without installing or running Ray. This provides just enough surface used by
the training entrypoints (init/is_initialized/remote/get/timeline/options).

Note: This shim is intentionally minimal. It does not provide parallelism or
distributed features; all calls execute synchronously in-process.
"""

from __future__ import annotations

from typing import Any, Callable, Type, TypeVar, Union

_initialized: bool = False


def is_initialized() -> bool:
    return _initialized


def init(*args, **kwargs) -> None:  # noqa: D401
    """Initialize the local shim. Accepts Ray-like kwargs and ignores them."""
    global _initialized
    _initialized = True


def timeline(filename: str | None = None) -> None:  # noqa: D401
    """No-op timeline writer for compatibility."""
    return None


def get(obj: Any) -> Any:
    """Mimic ray.get: unwrap values produced by .remote().

    - If obj is a list/tuple, return a list with each element passed through get.
    - If obj is a _RemoteResult, return its underlying value.
    - Otherwise, return obj as-is.
    """
    if isinstance(obj, (list, tuple)):
        return [get(x) for x in obj]
    if isinstance(obj, _RemoteResult):
        return obj.value
    return obj


def put(data: Any) -> Any:  # For limited compatibility where ray.put is used
    return data


T = TypeVar("T")


class _RemoteResult:
    def __init__(self, value: Any):
        self.value = value


class _MethodRemoteWrapper:
    def __init__(self, obj: Any, fn: Callable[..., Any]):
        self._obj = obj
        self._fn = fn

    def remote(self, *args: Any, **kwargs: Any) -> _RemoteResult:
        return _RemoteResult(self._fn(*args, **kwargs))


class _ActorHandle:
    def __init__(self, instance: Any):
        self._instance = instance

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._instance, name)
        if callable(attr):
            return _MethodRemoteWrapper(self._instance, attr)
        return attr


class _RemoteClassWrapper:
    def __init__(self, cls: Type[T]):
        self._cls = cls

    def options(self, *args: Any, **kwargs: Any) -> "_RemoteClassWrapper":
        # Options are ignored in the local shim
        return self

    def remote(self, *args: Any, **kwargs: Any) -> _ActorHandle:
        instance = self._cls(*args, **kwargs)
        return _ActorHandle(instance)


class _RemoteFuncWrapper:
    def __init__(self, fn: Callable[..., Any]):
        self._fn = fn

    def options(self, *args: Any, **kwargs: Any) -> "_RemoteFuncWrapper":
        return self

    def remote(self, *args: Any, **kwargs: Any) -> _RemoteResult:
        return _RemoteResult(self._fn(*args, **kwargs))


def remote(obj: Union[Callable[..., Any], Type[T]]):
    """Wrap a function or class with a Ray-like .remote/.options interface."""
    if isinstance(obj, type):
        return _RemoteClassWrapper(obj)
    if callable(obj):
        return _RemoteFuncWrapper(obj)
    raise TypeError("ray.remote expects a function or class")


# Minimal namespace for ray.util compatibility when imported elsewhere
class util:  # noqa: N801 - mimic ray.util
    @staticmethod
    def list_named_actors(*args: Any, **kwargs: Any):  # pragma: no cover - best-effort
        return []

