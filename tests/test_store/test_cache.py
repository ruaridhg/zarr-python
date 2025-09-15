from __future__ import annotations

import contextlib
import warnings
from collections import Counter
from typing import Any

import pytest

from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import cpu
from zarr.core.buffer.cpu import Buffer, buffer_prototype
from zarr.storage import LRUStoreCache, MemoryStore
from zarr.testing.store import StoreTests


class CounterStore(MemoryStore):  # type: ignore[misc]
    """
    A thin wrapper of MemoryStore to count different method calls for testing.
    """

    def __init__(self) -> None:
        super().__init__()
        self.counter: Counter[tuple[str, Any] | str] = Counter()

        # Add Store-like attributes that LRUStoreCache expects
        self.supports_writes = True
        self.supports_deletes = True
        self.supports_partial_writes = False
        self.supports_listing = True

    async def clear(self) -> None:
        self.counter["clear"] += 1
        # docstring inherited
        self._store_dict.clear()

    async def set(self, key: str, value: Any) -> None:
        """Store-like set method for async interface."""
        self.counter["set", key] += 1
        # Convert Buffer to bytes if needed
        if hasattr(value, "to_bytes"):
            self._store_dict[key] = value.to_bytes()
        else:
            self._store_dict[key] = value

    async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
        """Store-like get method for async interface."""
        self.counter["get", key] += 1
        try:
            data = self._store_dict[key]
            # Return as Buffer if prototype provided
            if prototype is not None and hasattr(prototype, "buffer"):
                from zarr.core.buffer.cpu import Buffer

                return Buffer.from_bytes(data)
            return data  # noqa: TRY300
        except KeyError:
            return None

    async def delete(self, key: str) -> None:
        """Store-like delete method for async interface."""
        self.counter["delete", key] += 1
        with contextlib.suppress(KeyError):
            del self._store_dict[key]

    async def exists(self, key: str) -> bool:
        """Store-like exists method for async interface."""
        self.counter["exists", key] += 1
        return key in self._store_dict

    async def getsize(self, key: str) -> int:
        """Store-like getsize method for async interface."""
        self.counter["getsize", key] += 1
        try:
            return len(self._store_dict[key])
        except KeyError:  # noqa: TRY203
            raise


class TestLRUStoreCache(StoreTests[LRUStoreCache, Buffer]):  # type: ignore[misc]
    store_cls = LRUStoreCache
    buffer_cls = cpu.Buffer
    CountingClass = CounterStore
    LRUStoreClass = LRUStoreCache
    root = ""

    async def get(self, store: LRUStoreCache, key: str) -> Buffer:
        """Get method required by StoreTests."""
        return await store.get(key, prototype=cpu.buffer_prototype)

    async def set(self, store: LRUStoreCache, key: str, value: Buffer) -> None:
        """Set method required by StoreTests."""
        await store.set(key, value)

    @pytest.fixture
    def store_kwargs(self) -> dict[str, Any]:
        """Provide default kwargs for store creation."""
        return {"store": MemoryStore(), "max_size": 2**27}

    @pytest.fixture
    async def store(self, store_kwargs: dict[str, Any]) -> LRUStoreCache:
        """Override store fixture to use constructor instead of open."""
        return self.store_cls(**store_kwargs)

    @pytest.fixture
    def open_kwargs(self) -> dict[str, Any]:
        """Provide default kwargs for store.open()."""
        return {"store": MemoryStore(), "max_size": 2**27}

    def create_store(self, **kwargs: Any) -> LRUStoreCache:
        return self.LRUStoreClass(MemoryStore(), max_size=2**27)

    def create_store_from_mapping(self, mapping: dict[str, Any], **kwargs: Any) -> LRUStoreCache:
        # Handle creation from existing mapping
        # Create a MemoryStore from the mapping
        underlying_store = MemoryStore()
        if mapping:
            # Convert mapping to store data
            for k, v in mapping.items():
                underlying_store._store_dict[k] = v
        return self.LRUStoreClass(underlying_store, max_size=2**27)

    async def test_cache_values_no_max_size(self) -> None:
        # setup store
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        await store.set(foo_key, b"xxx")
        await store.set(bar_key, b"yyy")
        assert 0 == store.counter["get", foo_key]
        assert 1 == store.counter["set", foo_key]
        assert 0 == store.counter["get", bar_key]
        assert 1 == store.counter["set", bar_key]

        # setup cache
        cache = self.LRUStoreClass(store, max_size=1024 * 1024)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first get(), cache miss
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["get", foo_key]
        assert 1 == store.counter["set", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second get(), cache hit
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["get", foo_key]  # No additional get call due to cache
        assert 1 == store.counter["set", foo_key]
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test set(), get()
        from zarr.core.buffer.cpu import Buffer

        await cache.set(foo_key, Buffer.from_bytes(b"zzz"))
        assert 1 == store.counter["get", foo_key]
        assert 2 == store.counter["set", foo_key]
        # should be a cache hit
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"zzz"
        assert 1 == store.counter["get", foo_key]  # No additional get call due to cache
        assert 2 == store.counter["set", foo_key]
        assert 2 == cache.hits
        assert 1 == cache.misses

        # manually invalidate all cached values
        cache.invalidate_values()
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"zzz"
        assert 2 == store.counter["get", foo_key]  # Cache invalidated, so new get call
        assert 2 == store.counter["set", foo_key]
        cache.invalidate()
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"zzz"
        assert 3 == store.counter["get", foo_key]  # Cache invalidated again, so another get call
        assert 2 == store.counter["set", foo_key]

        # test delete()
        await cache.delete(foo_key)
        result = await cache.get(foo_key)
        assert result is None
        # Verify the key is actually deleted from underlying store
        result = await store.get(foo_key)
        assert result is None

        # verify other keys untouched
        assert 0 == store.counter["get", bar_key]
        assert 1 == store.counter["set", bar_key]

    async def test_cache_values_with_max_size(self) -> None:
        # setup store
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        await store.set(foo_key, b"xxx")
        await store.set(bar_key, b"yyy")
        assert 0 == store.counter["get", foo_key]
        assert 0 == store.counter["get", bar_key]
        # setup cache - can only hold one item
        cache = self.LRUStoreClass(store, max_size=5)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' get(), cache miss
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["get", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' get(), cache hit
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["get", foo_key]  # No additional get call due to cache
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' get(), cache miss
        result = await cache.get(bar_key)
        assert result is not None
        assert result.to_bytes() == b"yyy"
        assert 1 == store.counter["get", bar_key]
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' get(), cache hit
        result = await cache.get(bar_key)
        assert result is not None
        assert result.to_bytes() == b"yyy"
        assert 1 == store.counter["get", bar_key]  # No additional get call due to cache
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' get(), should have been evicted, cache miss
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 2 == store.counter["get", foo_key]  # Cache miss due to eviction
        assert 2 == cache.hits
        assert 3 == cache.misses

        # test 'bar' get(), should have been evicted, cache miss
        result = await cache.get(bar_key)
        assert result is not None
        assert result.to_bytes() == b"yyy"
        assert 2 == store.counter["get", bar_key]  # Cache miss due to eviction
        assert 2 == cache.hits
        assert 4 == cache.misses

    async def test_cache_value_too_large_warning(self) -> None:
        """Test that a warning is emitted when a value is too large to cache."""
        # setup store with small cache
        store = self.CountingClass()
        foo_key = self.root + "foo"
        large_value = b"x" * 1000  # 1000 bytes
        small_cache_size = 500  # 500 bytes max cache

        await store.set(foo_key, large_value)
        cache = self.LRUStoreClass(store, max_size=small_cache_size)

        # Test that warning is emitted when trying to cache a value that's too large
        # This should trigger the warning since 1000 bytes > 500 bytes cache limit
        with pytest.warns(
            UserWarning, match=r"Value for key.*exceeds cache max_size.*and will not be cached"
        ):
            result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == large_value

        # Verify the value was not actually cached (cache miss on second access)
        assert cache.hits == 0  # No hits yet
        assert cache.misses == 1  # One miss from the first access

        # Second access should also be a miss since value wasn't cached
        # And it will also emit a warning, so we need to catch that too
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore warnings for this call
            result2 = await cache.get(foo_key)
            assert result2 is not None
            assert result2.to_bytes() == large_value
            assert cache.hits == 0  # Still no hits
            assert cache.misses == 2  # Two misses total

        # Verify the warning message contains expected information
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            await cache.get(foo_key)  # Trigger warning again

            assert len(w) == 1
            warning_message = str(w[0].message)
            assert f"Value for key '{foo_key}'" in warning_message
            assert "1,000 bytes" in warning_message  # Check formatted number
            assert "500 bytes" in warning_message  # Check cache size
            assert "Consider increasing max_size" in warning_message

    async def test_getsize_uses_cache(self) -> None:
        """Test that getsize() uses cached values when available."""
        store = self.CountingClass()
        cache = self.LRUStoreClass(store, max_size=1000)

        await store.set("key", b"value")

        # Populate cache
        await cache.get("key")
        assert 1 == store.counter["get", "key"]

        # getsize() should use cached value
        size = await cache.getsize("key")
        assert size == 5
        assert 1 == store.counter["get", "key"]  # No additional store access
        assert cache.hits == 1

    async def test_getsize_exception_handling(self) -> None:
        """Test that getsize() handles get() exceptions gracefully."""

        class FailingStore(CounterStore):
            async def get(self, key: str, prototype: Any = None, byte_range: Any = None) -> Any:
                if key == "fail":
                    raise RuntimeError("Simulated failure")
                return await super().get(key, prototype, byte_range)

        store = FailingStore()
        cache = self.LRUStoreClass(store, max_size=1000)
        await store.set("fail", b"x" * 50)  # Small value that would be cached

        # getsize() should work despite get() failing
        size = await cache.getsize("fail")
        assert size == 50
        assert cache.hits == 0  # No successful caching

    async def test_get_partial_values(self) -> None:
        """Test get_partial_values method."""

        # setup store
        store = MemoryStore()
        foo_key = "foo"
        bar_key = "bar"
        foo_data = b"hello world"
        bar_data = b"goodbye world"

        await store.set(foo_key, Buffer.from_bytes(foo_data))
        await store.set(bar_key, Buffer.from_bytes(bar_data))

        cache = self.LRUStoreClass(store, max_size=1024 * 1024)

        # Test getting partial values with byte ranges
        key_ranges = [
            (foo_key, RangeByteRequest(start=0, end=5)),  # "hello"
            (bar_key, RangeByteRequest(start=8, end=13)),  # "world"
            (foo_key, None),  # full value
        ]

        results = await cache.get_partial_values(buffer_prototype, key_ranges)

        assert len(results) == 3
        assert results[0] is not None
        assert results[0].to_bytes() == b"hello"
        assert results[1] is not None
        assert results[1].to_bytes() == b"world"
        assert results[2] is not None
        assert results[2].to_bytes() == foo_data

        # Test with non-existent key
        key_ranges_with_missing = [
            (foo_key, RangeByteRequest(start=0, end=5)),
            ("missing_key", None),
        ]

        results = await cache.get_partial_values(buffer_prototype, key_ranges_with_missing)
        assert len(results) == 2
        assert results[0] is not None
        assert results[0].to_bytes() == b"hello"
        assert results[1] is None

    async def test_set_partial_values(self) -> None:
        """Test set_partial_values method."""
        # setup store
        store = MemoryStore()
        cache = self.LRUStoreClass(store, max_size=1024 * 1024)

        key = "test_key"
        original_data = b"hello world 123"

        # Set initial data
        await cache.set(key, Buffer.from_bytes(original_data))

        # Test partial value setting
        partial_updates = [
            (key, 6, b"WORLD"),  # Replace "world" with "WORLD"
            (key, 12, b"456"),  # Replace "123" with "456"
        ]

        # Since MemoryStore doesn't implement set_partial_values,
        # it should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            await cache.set_partial_values(partial_updates)

        # The original data should still be in cache since the operation failed
        result = await cache.get(key)
        assert result is not None
        assert result.to_bytes() == original_data

        # Since the operation failed, there should be at least one cache hit
        # (from the initial set and then the get)
        assert cache.hits >= 1
