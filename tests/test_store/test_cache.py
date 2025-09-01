from __future__ import annotations

from collections import Counter
from typing import Any
import warnings

import pytest

from zarr.core.buffer import cpu
from zarr.core.buffer.cpu import Buffer
from zarr.storage import LRUStoreCache, MemoryStore
from zarr.testing.store import StoreTests


class CountingDict(dict[Any, Any]):
    """A dictionary that counts operations for testing purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.counter: Counter[tuple[str, Any] | str] = Counter()

    def __getitem__(self, key: Any) -> Any:
        self.counter["__getitem__", key] += 1
        return super().__getitem__(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        self.counter["__setitem__", key] += 1
        return super().__setitem__(key, value)

    def __contains__(self, key: Any) -> bool:
        self.counter["__contains__", key] += 1
        return super().__contains__(key)

    def __iter__(self) -> Any:
        self.counter["__iter__"] += 1
        return super().__iter__()

    def keys(self) -> Any:
        self.counter["keys"] += 1
        return super().keys()


def skip_if_nested_chunks(**kwargs: Any) -> None:
    if kwargs.get("dimension_separator") == "/":
        pytest.skip("nested chunks are unsupported")


class TestLRUStoreCache(StoreTests[LRUStoreCache, Buffer]):  # type: ignore[misc]
    store_cls = LRUStoreCache
    buffer_cls = cpu.Buffer
    CountingClass = CountingDict
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
        # wrapper therefore no dimension_separator argument
        skip_if_nested_chunks(**kwargs)
        return self.LRUStoreClass(MemoryStore(), max_size=2**27)

    def create_store_from_mapping(self, mapping: dict[str, Any], **kwargs: Any) -> LRUStoreCache:
        # Handle creation from existing mapping
        skip_if_nested_chunks(**kwargs)
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
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__getitem__", foo_key]
        assert 1 == store.counter["__setitem__", foo_key]
        assert 0 == store.counter["__getitem__", bar_key]
        assert 1 == store.counter["__setitem__", bar_key]

        # setup cache
        cache = self.LRUStoreClass(store, max_size=1024 * 1024)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first get(), cache miss
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == store.counter["__setitem__", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second get(), cache hit
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == store.counter["__setitem__", foo_key]
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test set(), get()
        from zarr.core.buffer.cpu import Buffer
        await cache.set(foo_key, Buffer.from_bytes(b"zzz"))
        assert 1 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]
        # should be a cache hit
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"zzz"
        assert 1 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]
        assert 2 == cache.hits
        assert 1 == cache.misses

        # manually invalidate all cached values
        cache.invalidate_values()
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"zzz"
        assert 2 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]
        cache.invalidate()
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"zzz"
        assert 3 == store.counter["__getitem__", foo_key]
        assert 2 == store.counter["__setitem__", foo_key]

        # test delete()
        await cache.delete(foo_key)
        result = await cache.get(foo_key)
        assert result is None
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            store[foo_key]

        # verify other keys untouched
        assert 0 == store.counter["__getitem__", bar_key]
        assert 1 == store.counter["__setitem__", bar_key]

    async def test_cache_values_with_max_size(self) -> None:
        # setup store
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__getitem__", foo_key]
        assert 0 == store.counter["__getitem__", bar_key]
        # setup cache - can only hold one item
        cache = self.LRUStoreClass(store, max_size=5)
        assert 0 == cache.hits
        assert 0 == cache.misses

        # test first 'foo' get(), cache miss
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["__getitem__", foo_key]
        assert 0 == cache.hits
        assert 1 == cache.misses

        # test second 'foo' get(), cache hit
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 1 == store.counter["__getitem__", foo_key]
        assert 1 == cache.hits
        assert 1 == cache.misses

        # test first 'bar' get(), cache miss
        result = await cache.get(bar_key)
        assert result is not None
        assert result.to_bytes() == b"yyy"
        assert 1 == store.counter["__getitem__", bar_key]
        assert 1 == cache.hits
        assert 2 == cache.misses

        # test second 'bar' get(), cache hit
        result = await cache.get(bar_key)
        assert result is not None
        assert result.to_bytes() == b"yyy"
        assert 1 == store.counter["__getitem__", bar_key]
        assert 2 == cache.hits
        assert 2 == cache.misses

        # test 'foo' get(), should have been evicted, cache miss
        result = await cache.get(foo_key)
        assert result is not None
        assert result.to_bytes() == b"xxx"
        assert 2 == store.counter["__getitem__", foo_key]
        assert 2 == cache.hits
        assert 3 == cache.misses

        # test 'bar' get(), should have been evicted, cache miss
        result = await cache.get(bar_key)
        assert result is not None
        assert result.to_bytes() == b"yyy"
        assert 2 == store.counter["__getitem__", bar_key]
        assert 2 == cache.hits
        assert 4 == cache.misses

    async def test_cache_keys(self) -> None:
        # setup
        store = self.CountingClass()
        foo_key = self.root + "foo"
        bar_key = self.root + "bar"
        baz_key = self.root + "baz"
        store[foo_key] = b"xxx"
        store[bar_key] = b"yyy"
        assert 0 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]
        assert 0 == store.counter["keys"]
        cache = self.LRUStoreClass(store, max_size=1024 * 1024)

        # keys should be cached on first call
        keys = sorted(cache.keys())
        assert keys == [bar_key, foo_key]
        assert 1 == store.counter["keys"]
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 1 == store.counter["keys"]
        assert foo_key in cache
        assert 1 == store.counter["__contains__", foo_key]
        # the next check for `foo_key` is cached
        assert foo_key in cache
        assert 1 == store.counter["__contains__", foo_key]
        assert keys == sorted(cache)
        assert 0 == store.counter["__iter__"]
        assert 1 == store.counter["keys"]

        # cache should be cleared if store is modified - crude but simple for now
        from zarr.core.buffer.cpu import Buffer
        await cache.set(baz_key, Buffer.from_bytes(b"zzz"))
        keys = sorted(cache.keys())
        assert keys == [bar_key, baz_key, foo_key]
        assert 2 == store.counter["keys"]
        # keys should now be cached
        assert keys == sorted(cache.keys())
        assert 2 == store.counter["keys"]

        # manually invalidate keys
        cache.invalidate_keys()
        keys = sorted(cache.keys())
        assert keys == [bar_key, baz_key, foo_key]
        assert 3 == store.counter["keys"]
        assert 1 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]
        cache.invalidate_keys()
        keys = sorted(cache)
        assert keys == [bar_key, baz_key, foo_key]
        assert 4 == store.counter["keys"]
        assert 1 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]
        cache.invalidate_keys()
        assert foo_key in cache
        assert 4 == store.counter["keys"]
        assert 2 == store.counter["__contains__", foo_key]
        assert 0 == store.counter["__iter__"]

        # check these would get counted if called directly
        assert foo_key in store
        assert 3 == store.counter["__contains__", foo_key]
        assert keys == sorted(store)
        assert 1 == store.counter["__iter__"]
        
    async def test_cache_value_too_large_warning(self) -> None:
        """Test that a warning is emitted when a value is too large to cache."""
        # setup store with small cache
        store = self.CountingClass()
        foo_key = self.root + "foo"
        large_value = b"x" * 1000  # 1000 bytes
        small_cache_size = 500  # 500 bytes max cache
        
        store[foo_key] = large_value
        cache = self.LRUStoreClass(store, max_size=small_cache_size)
        
        # Test that warning is emitted when trying to cache a value that's too large
        # This should trigger the warning since 1000 bytes > 500 bytes cache limit
        with pytest.warns(UserWarning, match=r"Value for key.*exceeds cache max_size.*and will not be cached"):
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
            assert "500 bytes" in warning_message    # Check cache size
            assert "Consider increasing max_size" in warning_message