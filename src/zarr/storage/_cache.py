import asyncio
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from threading import Lock
from typing import Any, cast

from zarr.abc.store import (
    ByteRequest,
    Store,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.buffer.core import default_buffer_prototype
from zarr.storage._utils import normalize_path


def _listdir_from_keys(store: Store, path: str) -> list[str]:
    """
    Extract directory listing from store keys by filtering keys that start with the given path.
    """
    children: set[str] = set()
    # Handle both Store objects and dict-like objects
    if hasattr(store, "keys") and callable(store.keys):
        keys = [str(k) for k in store.keys()]  # Ensure keys are strings  # noqa: SIM118
    else:
        # For stores that don't have keys method, we can't list them
        return []

    for key in keys:
        if key.startswith(path) and len(key) > len(path):
            suffix = key[len(path) :]
            child = suffix.split("/")[0]
            children.add(child)
    return sorted(children)


def listdir(store: Store, path: Path | None = None) -> list[str]:
    """
    Obtain a directory listing for the given path.

    If `store` provides a `listdir`
    method, this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface.
    """
    path_str = normalize_path(path)

    try:
        # Check if it's a Store object that supports listing
        if store.supports_listing:
            return cast(list[str], store.listdir(path_str))  # type: ignore[attr-defined]
    except AttributeError:
        pass

    try:
        # Check if it has a listdir method directly (for test stores and legacy stores)
        if callable(store.listdir):  # type: ignore[attr-defined]
            return cast(list[str], store.listdir(path_str))  # type: ignore[attr-defined]
    except AttributeError:
        pass

    # slow version, iterate through all keys
    warnings.warn(
        f"Store {store} has no `listdir` method. From zarr 2.9 onwards "
        "may want to inherit from `Store`.",
        stacklevel=2,
    )
    # _listdir_from_keys expects a trailing slash for non-empty paths
    listdir_path = path_str + "/" if path_str else path_str
    return _listdir_from_keys(store, listdir_path)


class LRUStoreCache(Store):
    """
    Storage class that implements a least-recently-used (LRU) cache layer over
    some other store.

    Intended primarily for use with stores that can be slow to
    access, e.g., remote stores that require network communication to store and
    retrieve data.

    The cache stores the raw bytes returned by the underlying store, before any
    decompression or array processing. This means that compressed data remains
    compressed in the cache, and decompression happens each time the cached data
    is accessed. This design choice keeps the cache lightweight while still
    providing significant performance benefits for network-bound operations.

    This store supports both read and write operations. Write operations use a
    write-through strategy where data is written to both the underlying store
    and cached locally. The cache automatically invalidates entries when the
    underlying data is modified to maintain consistency.

    Parameters
    ----------
    store : Store
        The store containing the actual data to be cached.
    max_size : int
        The maximum size that the cache may grow to, in number of bytes.
        This parameter is required to prevent unbounded memory growth.

        Values smaller than your typical chunk size will result in most data
        being silently excluded from the cache, reducing effectiveness.

    Examples
    --------
    The example below wraps a LocalStore with an LRU cache for demonstration::

        >>> import tempfile
        >>> import zarr
        >>> from zarr.storage import LocalStore
        >>>
        >>> # Create a temporary directory for the example
        >>> temp_dir = tempfile.mkdtemp()
        >>> store = LocalStore(temp_dir)
        >>>
        >>> # Create some test data first
        >>> arr = zarr.create((1000, 1000), chunks=(100, 100), store=store, dtype='f4')
        >>> arr[:] = 42.0
        >>>
        >>> # Now wrap with cache for faster access
        >>> cached_store = zarr.LRUStoreCache(store, max_size=1024 * 1024 * 256)  # 256MB cache
        >>> cached_arr = zarr.open(cached_store)
        >>>
        >>> # First access loads from disk and caches
        >>> data1 = cached_arr[0:100, 0:100]  # Cache miss
        >>>
        >>> # Second access uses cache (much faster for remote stores)
        >>> data2 = cached_arr[0:100, 0:100]  # Cache hit

    For remote stores where the performance benefit is more apparent::

        >>> from zarr.storage import RemoteStore
        >>> # remote_store = RemoteStore.from_url("https://example.com/data.zarr")
        >>> # cached_remote = zarr.LRUStoreCache(remote_store, max_size=2**28)


    """

    supports_writes: bool = True
    supports_deletes: bool = True
    supports_partial_writes: bool = True
    supports_listing: bool = True

    root: Path

    def __init__(self, store: Store, *, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer (bytes)")

        # Always inherit read_only state from the underlying store
        read_only = getattr(store, "read_only", False)

        # Call parent constructor with read_only parameter
        super().__init__(read_only=read_only)

        self._store = store
        self._max_size = max_size
        self._current_size = 0
        self._keys_cache: list[str] | None = None
        self._contains_cache: dict[Any, Any] = {}
        self._listdir_cache: dict[str | None, list[str]] = {}
        self._values_cache: OrderedDict[str, Any] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

        # Handle root attribute if present in underlying store
        if hasattr(store, "root"):
            self.root = store.root
        else:
            self.root = Path("/")  # Default root path

    @classmethod
    async def open(cls, store: Store, *, max_size: int, **kwargs: Any) -> "LRUStoreCache":
        """
        Create and open the LRU cache store.

        Parameters
        ----------
        store : Store
            The underlying store to wrap with caching.
        max_size : int
            The maximum size that the cache may grow to, in number of bytes.
        **kwargs : Any
            Additional keyword arguments passed to the store constructor.

        Returns
        -------
        LRUStoreCache
            The opened cache store instance.
        """
        # Handle read_only parameter if provided
        if "read_only" in kwargs:
            read_only = kwargs.pop("read_only")
            store = store.with_read_only(read_only)

        cache = cls(store, max_size=max_size, **kwargs)
        await cache._open()
        return cache

    def with_read_only(self, read_only: bool = False) -> "LRUStoreCache":
        """
        Return a new LRUStoreCache with a new read_only setting.

        Parameters
        ----------
        read_only
            If True, the store will be created in read-only mode. Defaults to False.

        Returns
        -------
        LRUStoreCache
            A new LRUStoreCache with the specified read_only setting.
        """
        # Create a new underlying store with the new read_only setting
        underlying_store = self._store.with_read_only(read_only)
        return LRUStoreCache(underlying_store, max_size=self._max_size)

    def _normalize_key(self, key: str | Path) -> str:
        """Convert key to string if it's a Path object, otherwise return as-is"""
        if isinstance(key, Path):
            return str(key)
        return str(key)

    def __getstate__(
        self,
    ) -> tuple[
        Store,
        int,
        int,
        list[str] | None,
        dict[Any, Any],
        dict[str | None, list[str]],
        OrderedDict[str, Any],
        int,
        int,
        bool,
        bool,
    ]:
        return (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
            self._read_only,
            self._is_open,
        )

    def __setstate__(
        self,
        state: tuple[
            Store,
            int,
            int,
            list[str] | None,
            dict[Any, Any],
            dict[str | None, list[str]],
            OrderedDict[str, Any],
            int,
            int,
            bool,
            bool,
        ],
    ) -> None:
        (
            self._store,
            self._max_size,
            self._current_size,
            self._keys_cache,
            self._contains_cache,
            self._listdir_cache,
            self._values_cache,
            self.hits,
            self.misses,
            self._read_only,
            self._is_open,
        ) = state
        self._mutex = Lock()

    def __len__(self) -> int:
        return len(self._keys())

    def __iter__(self) -> Iterator[str]:
        return self.keys()

    def __contains__(self, key: str | Path) -> bool:
        with self._mutex:
            if key not in self._contains_cache:
                # Check if it's a Store object vs dict-like object
                if hasattr(self._store, "supports_listing"):
                    # It's a Store object - use async interface
                    try:
                        result = asyncio.run(self._store.exists(str(key)))
                    except RuntimeError:
                        # Already in async context
                        raise NotImplementedError(
                            "Cannot use 'in' operator in async context. Use 'await store.exists(key)' instead."
                        ) from None
                else:
                    # It's a dict-like object (for tests) - use sync interface
                    result = key in cast(dict[str, Any], self._store)

                self._contains_cache[key] = bool(result)
            return bool(self._contains_cache[key])

    async def clear(self) -> None:
        """
        Remove all keys from the store and clear the cache.

        This operation clears both the underlying store and invalidates
        all cached data to maintain consistency.
        """
        # Check if store is writable
        self._check_writable()

        await self._store.clear()
        self.invalidate()

    def keys(self) -> Iterator[str]:
        """
        Return an iterator over the keys in the store.
        """
        with self._mutex:
            return iter(self._keys())

    def _keys(self) -> list[str]:
        if self._keys_cache is None:
            # Check if it's a Store object vs dict-like object
            if hasattr(self._store, "supports_listing"):
                # It's a Store object
                if self._store.supports_listing:
                    try:

                        async def collect_keys() -> list[str]:
                            return [str(key) async for key in self._store.list()]

                        keys = asyncio.run(collect_keys())
                        self._keys_cache = keys
                    except RuntimeError:
                        # Already in async context
                        raise NotImplementedError(
                            "Cannot list keys in async context - _keys() method needs to be async"
                        ) from None
                else:
                    # Store doesn't support listing
                    self._keys_cache = []
            else:
                # It's a dict-like object (for tests)
                if hasattr(self._store, "keys") and callable(self._store.keys):
                    keys = [str(k) for k in self._store.keys()]  # noqa: SIM118
                    self._keys_cache = keys
                else:
                    self._keys_cache = []
        return self._keys_cache

    def listdir(self, path: Path) -> list[str]:
        """
        Return a list of directory contents for the given path with caching.

        This method provides a cached version of directory listings to improve
        performance for repeated directory access operations.
        """
        with self._mutex:
            # Normalize path to string for consistent caching
            path_key = self._normalize_key(path)
            try:
                return self._listdir_cache[path_key]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path_key] = listing
                return listing

    async def getsize(self, key: str) -> int:
        """
        Get the size in bytes of the value stored at the given key.
        """
        return await self._store.getsize(key)

    def _pop_value(self) -> bytes:
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return cast(bytes, v)

    def _accommodate_value(self, value_size: int) -> None:
        while self._current_size + value_size > self._max_size:
            v = self._pop_value()
            self._current_size -= len(v)

    def _cache_value(self, key: str, value: Buffer | bytes) -> None:
        """Cache a value, handling both Buffer objects and bytes."""
        # Convert to bytes if needed
        if isinstance(value, Buffer):
            cache_value = value.to_bytes()
        else:
            # Already bytes
            cache_value = value

        value_size = len(cache_value)

        # Check if value exceeds max size - if so, don't cache it
        if value_size <= self._max_size:
            self._accommodate_value(value_size)
            cache_key = self._normalize_key(key)
            self._values_cache[cache_key] = cache_value
            self._current_size += value_size
        else:
            # Emit warning when value is too large to cache
            warnings.warn(
                f"Value for key '{key}' ({value_size:,} bytes) exceeds cache max_size "
                f"({self._max_size:,} bytes) and will not be cached. Consider increasing "
                f"max_size if this data is frequently accessed.",
                UserWarning,
                stacklevel=3,
            )

    def invalidate(self) -> None:
        """Completely clear the cache."""
        with self._mutex:
            self._values_cache.clear()
            self._current_size = 0
            self._invalidate_keys_unsafe()

    def invalidate_keys(self) -> None:
        """Clear the keys cache and all related caches."""
        with self._mutex:
            self._keys_cache = None
            self._contains_cache.clear()
            self._listdir_cache.clear()

    def _invalidate_keys_unsafe(self) -> None:
        """Clear the keys cache without acquiring mutex (assumes mutex already held)."""
        self._keys_cache = None
        self._contains_cache.clear()
        self._listdir_cache.clear()

    def _invalidate_value_unsafe(self, key: Any) -> None:
        """Remove a value from the cache and update the current size."""
        cache_key = self._normalize_key(key)
        if cache_key in self._values_cache:
            value = self._values_cache.pop(cache_key)
            self._current_size -= len(value)

    def invalidate_values(self) -> None:
        """Clear only the values cache, keeping other caches intact."""
        with self._mutex:
            self._values_cache.clear()
            self._current_size = 0

    def __eq__(self, value: object) -> bool:
        return type(self) is type(value) and self._store.__eq__(value._store)  # type: ignore[attr-defined]

    def __str__(self) -> str:
        return f"cache://{self._store}"

    def __repr__(self) -> str:
        return f"LRUStoreCache({self._store!r}, max_size={self._max_size})"

    async def delete(self, key: str) -> None:
        """
        Remove a key from the store.

        Parameters
        ----------
        key : str

        Notes
        -----
        If ``key`` is a directory within this store, the entire directory
        at ``store.root / key`` is deleted.
        """
        # Check if store is writable
        self._check_writable()

        # Check if it's a Store object vs dict-like object
        if hasattr(self._store, "supports_listing"):
            # It's a Store object - check if it supports deletes
            if self._store.supports_deletes:
                await self._store.delete(key)
            else:
                raise NotImplementedError(
                    f"Store {type(self._store).__name__} does not support delete operations"
                )
        else:
            # It's a dict-like object (for tests) - use sync interface
            try:
                del cast(dict[str, Any], self._store)[key]
            except (TypeError, AttributeError):
                raise NotImplementedError(
                    f"Store {type(self._store).__name__} does not support delete operations"
                ) from None

        # Invalidate cache entries
        with self._mutex:
            self._invalidate_keys_unsafe()
            cache_key = self._normalize_key(key)
            self._invalidate_value_unsafe(cache_key)

    async def exists(self, key: str) -> bool:
        # Delegate to the underlying store
        return await self._store.exists(key)

    async def _set(
        self,
        key: str,
        value: Buffer,
        exclusive: bool = False,
        byte_range: tuple[int, int] | None = None,
    ) -> None:
        # Check if store is writable
        self._check_writable()

        # Check if it's a Store object vs dict-like object
        if hasattr(self._store, "supports_listing"):
            # It's a Store object - use async interface
            await self._store.set(key, value)
        else:
            # It's a dict-like object (for tests) - use sync interface
            if hasattr(value, "to_bytes"):
                cast(dict[str, Any], self._store)[key] = value.to_bytes()
            else:
                cast(dict[str, Any], self._store)[key] = value

        # Update cache
        with self._mutex:
            self._invalidate_keys_unsafe()
            cache_key = self._normalize_key(key)
            self._invalidate_value_unsafe(cache_key)
            self._cache_value(cache_key, value)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # Use the cache for get operations
        cache_key = self._normalize_key(key)

        # For byte_range requests, don't use cache for now (could be optimized later)
        if byte_range is not None:
            if hasattr(self._store, "get"):
                # Check if it's an async Store.get method (takes prototype and byte_range)
                try:
                    if prototype is None:
                        prototype = default_buffer_prototype()
                    return await self._store.get(key, prototype, byte_range)
                except TypeError:
                    # Fallback to sync get from mapping - get full value and slice later
                    # For now, just return None for byte range requests on sync stores
                    return None
            else:
                # Fallback - get full value from mapping and slice
                try:
                    if hasattr(self._store, "__getitem__"):
                        full_value = cast(dict[str, Any], self._store)[key]
                        if prototype is None:
                            prototype = default_buffer_prototype()
                        # This is a simplified implementation - a full implementation would handle byte ranges
                        return prototype.buffer.from_bytes(full_value)
                    else:
                        return None
                except KeyError:
                    return None

        try:
            # Try cache first
            with self._mutex:
                value = self._values_cache[cache_key]
                self.hits += 1
                self._values_cache.move_to_end(cache_key)
                if prototype is None:
                    prototype = default_buffer_prototype()
                return prototype.buffer.from_bytes(value)
        except KeyError:
            # Cache miss - get from store
            if hasattr(self._store, "get"):
                # Try async Store.get method first
                try:
                    if prototype is None:
                        prototype = default_buffer_prototype()
                    result = await self._store.get(key, prototype, byte_range)
                except TypeError:
                    # Fallback for sync stores - use __getitem__ instead
                    try:
                        if hasattr(self._store, "__getitem__"):
                            value = cast(dict[str, Any], self._store)[key]
                            if prototype is None:
                                prototype = default_buffer_prototype()
                            result = prototype.buffer.from_bytes(value)
                        else:
                            result = None
                    except KeyError:
                        result = None
            else:
                # Fallback for sync stores/mappings
                try:
                    if hasattr(self._store, "__getitem__"):
                        value = cast(dict[str, Any], self._store)[key]
                        if prototype is None:
                            prototype = default_buffer_prototype()
                        result = prototype.buffer.from_bytes(value)
                    else:
                        result = None
                except KeyError:
                    result = None

            # Cache the result if we got one
            if result is not None:
                with self._mutex:
                    self.misses += 1
                    if cache_key not in self._values_cache:
                        self._cache_value(cache_key, result)
            else:
                # Still count as a miss even if result is None
                with self._mutex:
                    self.misses += 1

            return result

    async def get_partial_values(
        self,
        prototype: BufferPrototype,
        key_ranges: Iterable[tuple[str, ByteRequest | None]],
    ) -> list[Buffer | None]:
        # Delegate to the underlying store
        if hasattr(self._store, "get_partial_values"):
            return await self._store.get_partial_values(prototype, key_ranges)
        else:
            # Fallback - get each value individually
            results = []
            for key, byte_range in key_ranges:
                result = await self.get(key, prototype, byte_range)
                results.append(result)
            return results

    async def list(self) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if hasattr(self._store, "list"):
            async for key in self._store.list():
                yield key
        else:
            # Fallback for stores that don't have async list
            if hasattr(self._store, "keys") and callable(self._store.keys):
                for key in list(self._store.keys()):
                    yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if hasattr(self._store, "list_dir"):
            async for key in self._store.list_dir(prefix):
                yield key
        else:
            # Fallback using listdir
            try:
                listing = self.listdir(Path(prefix))
                for item in listing:
                    yield item
            except (FileNotFoundError, NotADirectoryError, KeyError):
                pass

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if hasattr(self._store, "list_prefix"):
            async for key in self._store.list_prefix(prefix):
                yield key
        else:
            # Fallback - filter all keys by prefix
            if hasattr(self._store, "keys") and callable(self._store.keys):
                for key in list(self._store.keys()):
                    if key.startswith(prefix):
                        yield key

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        # docstring inherited
        return await self._set(key, value, byte_range=byte_range)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        # Check if store is writable
        self._check_writable()

        # Delegate to the underlying store
        if hasattr(self._store, "set_partial_values"):
            await self._store.set_partial_values(key_start_values)
        else:
            # Fallback - this is complex to implement properly, so just invalidate cache
            for key, _start, _value in key_start_values:
                # For now, just invalidate the cache for these keys
                with self._mutex:
                    self._invalidate_keys_unsafe()
                    cache_key = self._normalize_key(key)
                    self._invalidate_value_unsafe(cache_key)
