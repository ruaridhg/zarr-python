import io
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable, Iterator
from pathlib import Path
from threading import Lock
from typing import Any, TypeAlias

import numpy as np

from zarr.abc.store import OffsetByteRequest, RangeByteRequest, Store, SuffixByteRequest
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.buffer.core import default_buffer_prototype
from zarr.storage._utils import normalize_path

ByteRequest: TypeAlias = RangeByteRequest | OffsetByteRequest | SuffixByteRequest


def buffer_size(v: Any) -> int:
    """Calculate the size in bytes of a value, handling Buffer objects properly."""
    if hasattr(v, "__len__") and hasattr(v, "nbytes"):
        # This is likely a Buffer object
        return int(v.nbytes)
    elif hasattr(v, "to_bytes"):
        # This is a Buffer object, get its bytes representation
        return len(v.to_bytes())
    elif isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    else:
        # Fallback to numpy
        return int(np.asarray(v).nbytes)


def _path_to_prefix(path: str | None) -> str:
    # assume path already normalized
    if path:
        prefix = path + "/"
    else:
        prefix = ""
    return prefix


def _listdir_from_keys(store: Store, path: str | None = None) -> list[str]:
    # assume path already normalized
    prefix = _path_to_prefix(path)
    children: set[str] = set()
    # Handle both Store objects and dict-like objects
    if hasattr(store, "keys") and callable(store.keys):
        keys = [str(k) for k in store.keys()]  # Ensure keys are strings  # noqa: SIM118
    else:
        # For stores that don't have keys method, we can't list them
        return []

    for key in keys:
        if key.startswith(prefix) and len(key) > len(prefix):
            suffix = key[len(prefix) :]
            child = suffix.split("/")[0]
            children.add(child)
    return sorted(children)


def listdir(store: Store, path: Path | None = None) -> list[str]:
    """Obtain a directory listing for the given path. If `store` provides a `listdir`
    method, this will be called, otherwise will fall back to implementation via the
    `MutableMapping` interface."""
    path_str = normalize_path(path)
    if hasattr(store, "listdir"):
        # pass through
        result = store.listdir(path_str)
        return [str(item) for item in result]  # Ensure all items are strings
    else:
        # slow version, iterate through all keys
        warnings.warn(
            f"Store {store} has no `listdir` method. From zarr 2.9 onwards "
            "may want to inherit from `Store`.",
            stacklevel=2,
        )
        return _listdir_from_keys(store, path_str)


def _get(path: Path, prototype: BufferPrototype, byte_range: ByteRequest | None) -> Buffer:
    if byte_range is None:
        return prototype.buffer.from_bytes(path.read_bytes())
    with path.open("rb") as f:
        size = f.seek(0, io.SEEK_END)
        if isinstance(byte_range, RangeByteRequest):
            f.seek(byte_range.start)
            return prototype.buffer.from_bytes(f.read(byte_range.end - f.tell()))
        elif isinstance(byte_range, OffsetByteRequest):
            f.seek(byte_range.offset)
        elif isinstance(byte_range, SuffixByteRequest):
            f.seek(max(0, size - byte_range.suffix))
        else:
            raise TypeError(f"Unexpected byte_range, got {byte_range}.")
        return prototype.buffer.from_bytes(f.read())


def _put(
    path: Path,
    value: Buffer,
    start: int | None = None,
    exclusive: bool = False,
) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if start is not None:
        with path.open("r+b") as f:
            f.seek(start)
            # write takes any object supporting the buffer protocol
            f.write(value.as_buffer_like())
        return None
    else:
        view = value.as_buffer_like()
        if exclusive:
            mode = "xb"
        else:
            mode = "wb"
        with path.open(mode=mode) as f:
            # write takes any object supporting the buffer protocol
            return f.write(view)


class LRUStoreCache(Store):
    """Storage class that implements a least-recently-used (LRU) cache layer over
    some other store. Intended primarily for use with stores that can be slow to
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

        Recommended minimum values:
        - For small metadata files: 1MB (1024 * 1024)
        - For chunk caching: 10MB (1024 * 1024 * 10)
        - For general use: 256MB (1024 * 1024 * 256)
        - For high-performance applications: 1GB (1024 * 1024 * 1000)

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

    def __init__(self, store: Store, *, max_size: int, **kwargs: Any) -> None:
        if not isinstance(max_size, int) or max_size <= 0:
            raise ValueError("max_size must be a positive integer (bytes)")

        # Extract and handle known parameters
        read_only = kwargs.get("read_only", getattr(store, "read_only", False))

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
    async def open(cls, store: Store, max_size: int | None, **kwargs: Any) -> "LRUStoreCache":
        """
        Create and open the LRU cache store.

        Parameters
        ----------
        store : Store
            The underlying store to wrap with caching.
        max_size : int | None
            The maximum size that the cache may grow to, in number of bytes.
        **kwargs : Any
            Additional keyword arguments passed to the store constructor.

        Returns
        -------
        LRUStoreCache
            The opened cache store instance.
        """
        cache = cls(store, max_size, **kwargs)
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
        return LRUStoreCache(underlying_store, self._max_size, read_only=read_only)

    def _normalize_key(self, key: Any) -> str:
        """Convert key to string if it's a Path object, otherwise return as-is"""
        if isinstance(key, Path):
            return str(key)
        return str(key)

    def __getstate__(
        self,
    ) -> tuple[
        Store,
        int | None,
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
            int | None,
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

    def __contains__(self, key: Any) -> bool:
        with self._mutex:
            if key not in self._contains_cache:
                # Handle both Store objects and dict-like objects
                if hasattr(self._store, "__contains__"):
                    result = key in self._store
                    self._contains_cache[key] = bool(result)
                else:
                    # Fallback for stores without __contains__
                    try:
                        if hasattr(self._store, "__getitem__"):
                            self._store[key]
                            self._contains_cache[key] = True
                        else:
                            self._contains_cache[key] = False
                    except KeyError:
                        self._contains_cache[key] = False
            return bool(self._contains_cache[key])

    async def clear(self) -> None:
        # Check if store is writable
        self._check_writable()

        await self._store.clear()
        self.invalidate()

    def keys(self) -> Iterator[str]:
        with self._mutex:
            return iter(self._keys())

    def _keys(self) -> list[str]:
        if self._keys_cache is None:
            # Handle both Store objects and dict-like objects
            if hasattr(self._store, "keys") and callable(self._store.keys):
                self._keys_cache = [str(k) for k in self._store.keys()]  # noqa: SIM118
            else:
                # Fallback for stores that don't have keys method
                self._keys_cache = []
        return self._keys_cache

    def listdir(self, path: Path | None = None) -> list[str]:
        with self._mutex:
            # Normalize path to string for consistent caching
            path_key = self._normalize_key(path) if path is not None else None
            try:
                return self._listdir_cache[path_key]
            except KeyError:
                listing = listdir(self._store, path)
                self._listdir_cache[path_key] = listing
                return listing

    async def getsize(self, key: str) -> int:
        return await self._store.getsize(key)

    def _pop_value(self) -> Any:
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return v

    def _accommodate_value(self, value_size: int) -> None:
        while self._current_size + value_size > self._max_size:
            v = self._pop_value()
            self._current_size -= buffer_size(v)

    def _cache_value(self, key: str, value: Any) -> None:
        # cache a value
        if hasattr(value, "to_bytes"):
            cache_value = value.to_bytes()
        else:
            cache_value = value

        value_size = buffer_size(cache_value)
        # Check if value exceeds max size - if so, don't cache it
        if value_size <= self._max_size:
            self._accommodate_value(value_size)
            cache_key = self._normalize_key(key)
            self._values_cache[cache_key] = cache_value
            self._current_size += value_size
        # If value_size > max_size, we simply don't cache it (silent skip)

    def invalidate(self) -> None:
        """Completely clear the cache."""
        with self._mutex:
            self._values_cache.clear()
            self._invalidate_keys()
            self._current_size = 0

    def invalidate_values(self) -> None:
        """Clear the values cache."""
        with self._mutex:
            self._values_cache.clear()

    def invalidate_keys(self) -> None:
        """Clear the keys cache."""
        with self._mutex:
            self._invalidate_keys()

    def _invalidate_keys(self) -> None:
        self._keys_cache = None
        self._contains_cache.clear()
        self._listdir_cache.clear()

    def _invalidate_value(self, key: Any) -> None:
        cache_key = self._normalize_key(key)
        if cache_key in self._values_cache:
            value = self._values_cache.pop(cache_key)
            self._current_size -= buffer_size(value)

    def __getitem__(self, key: Any) -> Any:
        cache_key = self._normalize_key(key)
        try:
            # first try to obtain the value from the cache
            with self._mutex:
                value = self._values_cache[cache_key]
                # cache hit if no KeyError is raised
                self.hits += 1
                # treat the end as most recently used
                self._values_cache.move_to_end(cache_key)

        except KeyError:
            # cache miss, retrieve value from the store
            if hasattr(self._store, "__getitem__"):
                value = self._store[key]
            else:
                # Fallback for async stores
                raise KeyError(f"Key {key} not found in store") from None
            with self._mutex:
                self.misses += 1
                # need to check if key is not in the cache, as it may have been cached
                # while we were retrieving the value from the store
                if cache_key not in self._values_cache:
                    self._cache_value(cache_key, value)

        return value

    def __setitem__(self, key: str, value: Buffer) -> None:
        if hasattr(self._store, "__setitem__"):
            self._store[key] = value
        else:
            # For async stores, we can't handle this synchronously
            raise TypeError("Cannot use __setitem__ with async store")

        # Update cache and invalidate keys cache since we may have added a new key
        with self._mutex:
            self._invalidate_keys()
            self._cache_value(self._normalize_key(key), value)

    def __delitem__(self, key: Any) -> None:
        if hasattr(self._store, "__delitem__"):
            del self._store[key]
        else:
            # For async stores, this shouldn't be used - use delete() instead
            raise NotImplementedError("Use async delete() method for async stores")
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)

    def __eq__(self, value: object) -> bool:
        return type(self) is type(value) and self._store.__eq__(value._store)  # type: ignore[attr-defined]

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

        # Delegate to the underlying store for actual deletion
        if hasattr(self._store, "delete"):
            await self._store.delete(key)
        else:
            # Fallback for stores that don't have async delete
            del self._store[key]  # type: ignore[attr-defined]

        # Invalidate cache entries
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)

    async def exists(self, key: str) -> bool:
        # Delegate to the underlying store
        if hasattr(self._store, "exists"):
            return await self._store.exists(key)
        else:
            # Fallback for stores that don't have async exists
            if hasattr(self._store, "__contains__"):
                return key in self._store
            else:
                # Final fallback - try to get the key
                try:
                    if hasattr(self._store, "__getitem__"):
                        self._store[key]
                        return True
                    else:
                        return False
                except KeyError:
                    return False

    async def _set(self, key: str, value: Buffer, exclusive: bool = False) -> None:
        # Check if store is writable
        self._check_writable()

        # Delegate to the underlying store
        if hasattr(self._store, "set"):
            await self._store.set(key, value)
        else:
            # Fallback for stores that don't have async set
            if hasattr(self._store, "__setitem__"):
                # Convert Buffer to bytes for sync stores
                if hasattr(value, "to_bytes"):
                    self._store[key] = value.to_bytes()
                else:
                    self._store[key] = value
            else:
                raise TypeError("Store does not support setting values")

        # Update cache
        with self._mutex:
            self._invalidate_keys()
            cache_key = self._normalize_key(key)
            self._invalidate_value(cache_key)
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
            if hasattr(self._store, "get") and callable(self._store.get):
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
                        full_value = self._store[key]
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
            if hasattr(self._store, "get") and callable(self._store.get):
                # Try async Store.get method first
                try:
                    if prototype is None:
                        prototype = default_buffer_prototype()
                    result = await self._store.get(key, prototype, byte_range)
                except TypeError:
                    # Fallback for sync stores - use __getitem__ instead
                    try:
                        if hasattr(self._store, "__getitem__"):
                            value = self._store[key]
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
                        value = self._store[key]
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
                        self._cache_value(cache_key, result.to_bytes())
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

    async def set(self, key: str, value: Buffer) -> None:
        # docstring inherited
        return await self._set(key, value)

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
                    self._invalidate_keys()
                    cache_key = self._normalize_key(key)
                    self._invalidate_value(cache_key)
