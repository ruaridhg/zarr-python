from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from zarr.abc.store import RangeByteRequest
from zarr.core.buffer import cpu
from zarr.core.buffer.cpu import Buffer, buffer_prototype
from zarr.storage import LRUStoreCache, MemoryStore
from zarr.storage._cache import _listdir_from_keys
from zarr.testing.store import StoreTests


class CountingStore:
    def __init__(self) -> None:
        self.listdir_call_count = 0
        self._directories = {
            "root": ["file1.txt", "file2.txt", "subdir"],
            "root/subdir": ["file3.txt", "file4.txt", "nested"],
            "root/subdir/nested": ["file5.txt"],
            "nonexistent": [],
        }

    def listdir(self, path: str) -> list[str]:
        """Implement listdir for testing purposes."""
        self.listdir_call_count += 1
        return self._directories.get(path, [])


class DictLikeStore:
    """A simple dict-like store for testing _listdir_from_keys."""

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def keys(self) -> list[str]:
        return self._keys


class StoreWithoutKeys:
    """A store without a keys method for testing _listdir_from_keys."""


class StoreWithNonCallableKeys:
    """A store with keys attribute that is not callable for testing _listdir_from_keys."""

    keys = "not_callable"


class StoreWithNonStringKeys:
    """A store with mixed key types for testing _listdir_from_keys type conversion."""

    def keys(self) -> list[Any]:
        return [b"root/file1.txt", 123, "root/file2.txt"]


class StoreWithMixedConvertibleKeys:
    """A store with string keys for testing _listdir_from_keys."""

    def keys(self) -> list[Any]:
        return ["root/file1.txt", "root/file2.txt", "123/file3.txt"]


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
    CountingClassStore = CountingStore
    DictLikeClass = DictLikeStore
    StoreWithoutKeysClass = StoreWithoutKeys
    StoreWithNonCallableKeysClass = StoreWithNonCallableKeys
    StoreWithNonStringKeysClass = StoreWithNonStringKeys
    StoreWithMixedConvertibleKeysClass = StoreWithMixedConvertibleKeys
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

    async def test_listdir(self) -> None:
        """Test listdir method with caching."""
        store = self.CountingClassStore()
        cache = self.LRUStoreClass(store, max_size=1024 * 1024)

        # Test listdir on root directory
        root_listing = cache.listdir(Path("root"))
        expected_root = ["file1.txt", "file2.txt", "subdir"]
        assert sorted(root_listing) == sorted(expected_root)
        assert store.listdir_call_count == 1

        # Second call should be cached (no additional call to underlying store)
        root_listing2 = cache.listdir(Path("root"))
        assert sorted(root_listing2) == sorted(expected_root)
        assert store.listdir_call_count == 1  # Still 1, cached

        # Test listdir on subdirectory
        subdir_listing = cache.listdir(Path("root/subdir"))
        expected_subdir = ["file3.txt", "file4.txt", "nested"]
        assert sorted(subdir_listing) == sorted(expected_subdir)
        assert store.listdir_call_count == 2  # New call

        # Test cached subdirectory listing
        subdir_listing2 = cache.listdir(Path("root/subdir"))
        assert sorted(subdir_listing2) == sorted(expected_subdir)
        assert store.listdir_call_count == 2  # Still cached

        # Test listdir on nested directory
        nested_listing = cache.listdir(Path("root/subdir/nested"))
        expected_nested = ["file5.txt"]
        assert sorted(nested_listing) == sorted(expected_nested)
        assert store.listdir_call_count == 3

        # Test listdir on non-existent directory
        empty_listing = cache.listdir(Path("nonexistent"))
        assert empty_listing == []
        assert store.listdir_call_count == 4

        # Test cache invalidation by manually clearing cache
        cache.invalidate_keys()

        # Cache should be invalidated, so this should make a new call
        root_listing3 = cache.listdir(Path("root"))
        assert sorted(root_listing3) == sorted(expected_root)
        assert store.listdir_call_count == 5  # New call due to invalidation

        # Test the else branch - store without listdir method
        keys = ["root/file1.txt", "root/file2.txt", "root/subdir/file3.txt", "other/file4.txt"]
        dict_store = self.DictLikeClass(keys)
        cache_dict = LRUStoreCache(dict_store, max_size=100)

        # This should trigger the else branch and call _listdir_from_keys
        with pytest.warns(UserWarning, match="Store.*has no.*listdir.*method"):
            result = cache_dict.listdir(Path("root"))
        expected = ["file1.txt", "file2.txt", "subdir"]
        assert sorted(result) == sorted(expected)

    def test_listdir_from_keys(self) -> None:
        """Test the _listdir_from_keys function with various scenarios."""

        # Test basic directory structure
        keys = [
            "root/file1.txt",
            "root/file2.txt",
            "root/subdir/file3.txt",
            "root/subdir/nested/file4.txt",
            "other/file5.txt",
        ]
        store = self.DictLikeClass(keys)

        # Test listing root directory
        result = _listdir_from_keys(store, "root/")
        expected = ["file1.txt", "file2.txt", "subdir"]
        assert sorted(result) == sorted(expected)

        # Test listing subdirectory
        result = _listdir_from_keys(store, "root/subdir/")
        expected = ["file3.txt", "nested"]
        assert sorted(result) == sorted(expected)

        # Test listing nested directory
        result = _listdir_from_keys(store, "root/subdir/nested/")
        expected = ["file4.txt"]
        assert sorted(result) == sorted(expected)

        # Test listing non-existent directory
        result = _listdir_from_keys(store, "nonexistent/")
        assert result == []

        # Test with empty path (should return top-level items)
        result = _listdir_from_keys(store, "")
        expected = ["root", "other"]
        assert sorted(result) == sorted(expected)

        # Test with path that matches a file exactly (should return empty)
        result = _listdir_from_keys(store, "root/file1.txt")
        assert result == []

        # Test with store that has no keys method
        store_no_keys = self.StoreWithoutKeysClass()
        result = _listdir_from_keys(store_no_keys, "root/")
        assert result == []

        # Test with store that has keys but it's not callable
        store_non_callable = self.StoreWithNonCallableKeysClass()
        result = _listdir_from_keys(store_non_callable, "root/")
        assert result == []

        # Test edge case: keys that don't start with path
        keys_no_match = ["different/file1.txt", "another/file2.txt"]
        store_no_match = self.DictLikeClass(keys_no_match)
        result = _listdir_from_keys(store_no_match, "root/")
        assert result == []

        # Test edge case: path equals key length (should be excluded)
        keys_equal_length = ["root/", "root/file.txt"]
        store_equal = self.DictLikeClass(keys_equal_length)
        result = _listdir_from_keys(store_equal, "root/")
        expected = ["file.txt"]
        assert result == expected

        # Test with non-string keys (should be converted to strings)
        store_mixed = self.StoreWithNonStringKeysClass()
        result = _listdir_from_keys(store_mixed, "root/")
        # Should handle conversion to strings - only the actual string key will match
        expected = ["file2.txt"]  # Only the string key "root/file2.txt" matches
        assert sorted(result) == sorted(expected)

        # Test with proper string keys to ensure bytes and int conversions don't match
        store_mixed_proper = self.StoreWithMixedConvertibleKeysClass()
        result = _listdir_from_keys(store_mixed_proper, "root/")
        expected = ["file1.txt", "file2.txt"]
        assert sorted(result) == sorted(expected)
