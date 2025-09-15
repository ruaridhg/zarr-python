"""
Test the max_size functionality of CacheStore
"""
import zarr
import numpy as np
from zarr.storage._caching_store import CacheStore
from zarr.storage import MemoryStore, LocalStore

print("=== Testing CacheStore max_size functionality ===")

# Create stores
source_store = MemoryStore()
cache_store = MemoryStore()

# Create a CacheStore with a small max_size (1KB)
cached_store = CacheStore(
    source_store, 
    cache_store=cache_store, 
    max_size=1024,  # 1KB limit
    key_insert_times={}
)

print(f"Initial cache info: {cached_store.cache_info()}")

# Create some test data that will exceed the cache size
print("\nCreating test arrays...")
z1 = zarr.zeros((50, 50), chunks=(50, 50), dtype='f4', store=cached_store, path='array1')
z2 = zarr.zeros((50, 50), chunks=(50, 50), dtype='f4', store=cached_store, path='array2') 
z3 = zarr.zeros((50, 50), chunks=(50, 50), dtype='f4', store=cached_store, path='array3')

# Fill arrays with data (each chunk is 50*50*4 = 10KB, larger than our 1KB cache)
print("Filling arrays with data...")
z1[:] = np.random.random((50, 50))
z2[:] = np.random.random((50, 50)) + 1
z3[:] = np.random.random((50, 50)) + 2

print(f"After creating arrays: {cached_store.cache_info()}")

# Read data - this should trigger cache eviction due to size limits
print("\nReading array1 (should cache and potentially evict due to size)...")
data1 = z1[:]
print(f"After reading array1: {cached_store.cache_info()}")

print("\nReading array2 (should cache and potentially evict array1)...")
data2 = z2[:]
print(f"After reading array2: {cached_store.cache_info()}")

print("\nReading array3 (should cache and potentially evict array2)...")
data3 = z3[:]
print(f"After reading array3: {cached_store.cache_info()}")

# Test without max_size for comparison
print("\n=== Testing without max_size limit ===")
unlimited_cached_store = CacheStore(
    source_store, 
    cache_store=MemoryStore(),  # Fresh cache store
    max_size=None,  # No size limit
    key_insert_times={}
)

print(f"Unlimited cache initial info: {unlimited_cached_store.cache_info()}")

# Create arrays with unlimited cache
z1_unlimited = zarr.open(unlimited_cached_store, path='array1')
z2_unlimited = zarr.open(unlimited_cached_store, path='array2')
z3_unlimited = zarr.open(unlimited_cached_store, path='array3')

print("Reading all arrays with unlimited cache...")
data1_unlimited = z1_unlimited[:]
print(f"After reading array1: {unlimited_cached_store.cache_info()}")

data2_unlimited = z2_unlimited[:]
print(f"After reading array2: {unlimited_cached_store.cache_info()}")

data3_unlimited = z3_unlimited[:]
print(f"After reading array3: {unlimited_cached_store.cache_info()}")

print("\n=== Comparison ===")
print(f"Limited cache (1KB):   {cached_store.cache_info()}")
print(f"Unlimited cache:       {unlimited_cached_store.cache_info()}")

print("\n=== Data consistency check ===")
print(f"Array1 data matches: {np.array_equal(data1, data1_unlimited)}")
print(f"Array2 data matches: {np.array_equal(data2, data2_unlimited)}")
print(f"Array3 data matches: {np.array_equal(data3, data3_unlimited)}")

print("\n=== Test clearing cache ===")
print("Note: clear_cache is async, so we'll skip this test in the sync script")
print(f"Current cache info: {unlimited_cached_store.cache_info()}")

print("\n=== Test complete ===")
