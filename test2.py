import zarr
from zarr.storage._caching_store import CacheStore
from zarr.storage._memory import MemoryStore

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