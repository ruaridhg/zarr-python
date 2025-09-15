import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterable
from threading import Lock
from typing import Any, Self

from zarr.abc.store import (
    ByteRequest,
    Store,
)
from zarr.core.buffer import Buffer, BufferPrototype
from zarr.core.buffer.core import default_buffer_prototype


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

    @property
    def supports_writes(self) -> bool:
        """Whether the underlying store supports write operations."""
        return self._store.supports_writes

    @property
    def supports_deletes(self) -> bool:
        """Whether the underlying store supports delete operations."""
        return self._store.supports_deletes

    @property
    def supports_partial_writes(self) -> bool:
        """Whether the underlying store supports partial write operations."""
        return self._store.supports_partial_writes

    @property
    def supports_listing(self) -> bool:
        """Whether the underlying store supports listing operations."""
        return self._store.supports_listing

    def __init__(self, store: Store, *, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("max_size must be a positive integer (bytes)")

        # Always inherit read_only state from the underlying store
        read_only = store.read_only
        super().__init__(read_only=read_only)

        self._store = store
        self._max_size = max_size
        self._current_size = 0
        self._contains_cache: dict[Any, Any] = {}
        self._listdir_cache: dict[str | None, list[str]] = {}
        self._values_cache: OrderedDict[str, bytes] = OrderedDict()
        self._mutex = Lock()
        self.hits = self.misses = 0

    @classmethod
    async def open(cls, store: Store, *, max_size: int, read_only: bool = False) -> Self:
        """
        Create and open a new LRU cache store.

        Parameters
        ----------
        store : Store
            The underlying store to wrap with caching.
        max_size : int
            The maximum size that the cache may grow to, in number of bytes.

        Returns
        -------
        LRUStoreCache
            The opened cache store instance.
        """

        cache = cls(store, max_size=max_size)

        if read_only:
            cache._read_only = True
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

    def __getstate__(
        self,
    ) -> tuple[
        Store,
        int,
        int,
        dict[Any, Any],
        dict[str | None, list[str]],
        OrderedDict[str, bytes],
        int,
        int,
        bool,
        bool,
    ]:
        return (
            self._store,
            self._max_size,
            self._current_size,
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
            dict[Any, Any],
            dict[str | None, list[str]],
            OrderedDict[str, bytes],
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
        return len(self._values_cache)

    async def clear(self) -> None:
        """
        Remove all keys from the store and clear the cache.

        This operation clears both the underlying store and invalidates
        all cached data to maintain consistency.
        """

        await self._store.clear()
        self.invalidate()

    async def getsize(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
    ) -> int:
        """
        Get the size in bytes of the value stored at the given key.

        For remote stores, this method attempts to get and cache the value
        since network latency typically dominates the cost of both getsize()
        and get() operations, making it more efficient to retrieve the full
        value during size queries.
        """
        cache_key = key

        # Check cache first
        with self._mutex:
            if cache_key in self._values_cache:
                cached_value = self._values_cache[cache_key]
                # Move to end to mark as recently used
                self._values_cache.move_to_end(cache_key)
                self.hits +=1
                return len(cached_value)

        # Not in cache, delegate to underlying store
        self.misses += 1

        if prototype is None:
            prototype = default_buffer_prototype()

        # Try to get the full value first (better for remote stores)
        try:
            value = await self._store.get(key, prototype)
            if value is not None:
                # Successfully got the value, cache it and return size
                with self._mutex:
                    if cache_key not in self._values_cache:
                        self._cache_value(cache_key, value)

                # Return size based on the actual value we retrieved
                return len(value)
        except (KeyError, FileNotFoundError):
            pass
        except NotImplementedError:
            pass
        except (ConnectionError, TimeoutError, RuntimeError):
            pass
        except Exception:
            # Re-raise unexpected exceptions rather than silently falling back
            raise

        # Fallback to underlying store's getsize() method
        return await self._store.getsize(key)

    def _pop_value(self) -> bytes:
        # remove the first value from the cache, as this will be the least recently
        # used value
        _, v = self._values_cache.popitem(last=False)
        return v

    def _accommodate_value(self, value_size: int) -> None:
        # Remove items from the cache until there's enough room for a value
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
            cache_key = key
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

        if self._store.supports_deletes:
            await self._store.delete(key)
        else:
            raise NotImplementedError(
                f"Store {type(self._store).__name__} does not support delete operations"
            )

        # Invalidate cache entries
        self.invalidate_values()

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists in the store.

        This method first checks the cache for the key to avoid
        unnecessary calls to the underlying store.
        """
        cache_key = key

        # Check cache first
        with self._mutex:
            if cache_key in self._values_cache:
                # Key exists in cache, so it exists in store
                # Move to end to mark as recently used
                self._values_cache.move_to_end(cache_key)
                return True

        # Not in cache, delegate to underlying store
        return await self._store.exists(key)

    async def get(
        self,
        key: str,
        prototype: BufferPrototype | None = None,
        byte_range: ByteRequest | None = None,
    ) -> Buffer | None:
        # Use the cache for get operations
        cache_key = key

        if prototype is None:
            prototype = default_buffer_prototype()

        # For byte_range requests, don't use cache for now (could be optimized later)
        if byte_range is not None:
            return await self._store.get(key, prototype, byte_range)

        if cache_key in self._values_cache:
            # Try cache first
            with self._mutex:
                value = self._values_cache[cache_key]
                self.hits += 1
                self._values_cache.move_to_end(cache_key)
                return prototype.buffer.from_bytes(value)
        else:
            # Cache miss - get from store
            result = await self._store.get(key, prototype, byte_range)

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
        if self.supports_partial_writes:
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
        if self.supports_listing:
            async for key in self._store.list():
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if self.supports_listing:
            async for key in self._store.list_dir(prefix):
                yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        # Delegate to the underlying store
        if self.supports_listing:
            async for key in self._store.list_prefix(prefix):
                yield key

    async def set(self, key: str, value: Buffer, byte_range: tuple[int, int] | None = None) -> None:
        # docstring inherited
        # Check if store is writable
        self._check_writable()

        # Write to underlying store first
        await self._store.set(key, value)

        # Update cache with the new value
        cache_key = key
        with self._mutex:
            if cache_key in self._values_cache:
                old_value = self._values_cache[cache_key]
                self._current_size -= len(old_value)
                del self._values_cache[cache_key]

            # Cache the new value
            self._cache_value(cache_key, value)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, bytes | bytearray | memoryview]]
    ) -> None:
        # Check if store is writable
        self._check_writable()

        # Delegate to the underlying store
        if self.supports_partial_writes:
            await self._store.set_partial_values(key_start_values)
        else:
            # Fallback - this is complex to implement properly, so just invalidate cache
            for _key, _start, _value in key_start_values:
                # For now, just invalidate the cache for these keys
                self.invalidate_values()
