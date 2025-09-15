# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "zarr @ git+https://github.com/zarr-developers/zarr-python.git@main",
# ]
# ///
#

import time
import zarr
from typing_extensions import Literal
from zarr.abc.store import ByteRequest, Store
from zarr.core.buffer.core import Buffer, BufferPrototype
from zarr.storage._memory import MemoryStore
from zarr.storage._wrapper import WrapperStore
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStore(WrapperStore[Store]):
    _cache: Store
    max_age_seconds: int | Literal["infinity"]
    key_insert_times: dict[str, float]
    cache_set_data: bool

    def __init__(self, store: Store, *, cache_store: Store, max_age_seconds: int | Literal["infinity"] = "infinity", key_insert_times: dict[str, float] | None = None, cache_set_data: bool = True) -> None:
        super().__init__(store)
        self._cache = cache_store
        self.max_age_seconds = max_age_seconds
        if key_insert_times is None:
            key_insert_times = {}
        else:
            self.key_insert_times = key_insert_times
        self.cache_set_data = cache_set_data

    def _is_key_fresh(self, key: str) -> bool:
        if self.max_age_seconds == "infinity":
            return True
        else:
            now = time.monotonic()
            elapsed = now - self.key_insert_times.get(key, 0)
            return elapsed < self.max_age_seconds

    async def _get_try_cache(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        maybe_cached_result = await self._cache.get(key, prototype, byte_range)
        if maybe_cached_result is not None:
            logger.info('_get_try_cache: key %s found in cache', key)
            return maybe_cached_result
        else:
            logger.info('_get_try_cache: key %s not found in cache, fetching from store', key)
            maybe_fresh_result = await super().get(key, prototype, byte_range)
            if maybe_fresh_result is None:
                await self._cache.delete(key)
            else:
                await self._cache.set(key, maybe_fresh_result)
            return maybe_fresh_result

    async def _get_no_cache(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        maybe_fresh_result = await super().get(key, prototype, byte_range)
        if maybe_fresh_result is None:
            logger.info('_get_no_cache: key %s not found in store, deleting from cache', key)
            await self._cache.delete(key)
            self.key_insert_times.pop(key, None)
        else:
            logger.info('_get_no_cache: key %s found in store, setting in cache', key)
            await self._cache.set(key, maybe_fresh_result)
            self.key_insert_times[key] = time.monotonic()
        return maybe_fresh_result

    async def get(self, key: str, prototype: BufferPrototype, byte_range: ByteRequest | None = None) -> Buffer | None:
        if self._is_key_fresh(key):
            logger.info('get: key %s is not fresh, fetching from store', key)
            return await self._get_no_cache(key, prototype, byte_range)
        else:
            logger.info('get: key %s is fresh, trying cache', key)
            return await self._get_try_cache(key, prototype, byte_range)

    async def set(self, key: str, value: Buffer) -> None:
        logger.info('set: setting key %s in store', key)
        await super().set(key, value)
        if self.cache_set_data:
            logger.info('set: setting key %s in cache', key)
            await self._cache.set(key, value)
            self.key_insert_times[key] = time.monotonic()
        else:
            logger.info('set: deleting key %s from cache', key)
            await self._cache.delete(key)
            self.key_insert_times.pop(key, None)


    async def delete(self, key: str) -> None:
        logger.info('delete: deleting key %s from store', key)
        await super().delete(key)
        logger.info('delete: deleting key %s from cache', key)
        await self._cache.delete(key)
        self.key_insert_times.pop(key, None)

store_a = MemoryStore({})
store_b = MemoryStore({})

cached_store = CacheStore(store=store_a, cache_store=store_b, max_age_seconds=10, key_insert_times={})

z = zarr.create_array(cached_store, shape=(8,), dtype='float32', chunks=(4,))

print('filling array with values')
z[:] = 10

print('fetching value from first chunk')
z[0]

print('fetching value from first chunk again')
z[0]

print('erasing arrays values')
z[:] = z.fill_value

print('fetching value from first chunk')
z[0]

print('fetching value from first chunk again')
z[0]

"""
INFO:__main__:get: key zarr.json is stale, fetching from store
INFO:__main__:_get_no_cache: key zarr.json not found in store, deleting from cache
INFO:__main__:set: setting key zarr.json in store
INFO:__main__:set: setting key zarr.json in cache
filling array with values
INFO:__main__:set: setting key c/0 in store
INFO:__main__:set: setting key c/0 in cache
INFO:__main__:set: setting key c/1 in store
INFO:__main__:set: setting key c/1 in cache
fetching value from first chunk
INFO:__main__:get: key c/0 is fresh, trying cache
INFO:__main__:_get_try_cache: key c/0 found in cache
fetching value from first chunk again
INFO:__main__:get: key c/0 is fresh, trying cache
INFO:__main__:_get_try_cache: key c/0 found in cache
erasing arrays values
INFO:__main__:delete: deleting key c/0 from store
INFO:__main__:delete: deleting key c/0 from cache
INFO:__main__:delete: deleting key c/1 from store
INFO:__main__:delete: deleting key c/1 from cache
fetching value from first chunk
INFO:__main__:get: key c/0 is stale, fetching from store
INFO:__main__:_get_no_cache: key c/0 not found in store, deleting from cache
fetching value from first chunk again
INFO:__main__:get: key c/0 is stale, fetching from store
INFO:__main__:_get_no_cache: key c/0 not found in store, deleting from cache
"""