import zarr
import time
import numpy as np
import os
from zarr.storage._caching_store import CacheStore
from zarr.storage import MemoryStore, LocalStore

print("=== CacheStore Performance Test ===")

# Set up stores like test2.py
store_a = LocalStore('test3.zarr')  # Use LocalStore instead of MemoryStore for persistence
store_b = MemoryStore({})  # Cache backend

# Create cached store with the same pattern as test2.py
cached_store = CacheStore(store=store_a, cache_store=store_b, max_age_seconds=10, key_insert_times={})

# Create array using the cached store
print('Creating zarr array with cached store...')
z_cached = zarr.create_array(cached_store, shape=(100, 100), dtype='float32', chunks=(10, 10), mode='w')

print('Filling array with values...')
z_cached[:] = np.random.random((100, 100))  # Fill with random data like test.py

# Ensure the data is written to disk
print(f"Chunks created in test3.zarr: {os.listdir('test3.zarr') if os.path.exists('test3.zarr') else 'Directory not found'}")
if os.path.exists('test3.zarr') and 'c' in os.listdir('test3.zarr'):
    chunk_files = os.listdir('test3.zarr/c')
    print(f"Number of chunk files: {len(chunk_files)}")

# Performance comparison like test.py
print("\n=== Performance Benchmarks ===")

# Benchmark 1: Multiple reads with cache
print("Benchmarking reads with CacheStore...")
start = time.time()
for i in range(50):
    print(f'Reading iteration {i+1}/50 with cache...')
    _ = z_cached[:]  # Read entire array
elapsed_cache = time.time() - start

# Benchmark 2: Same reads without cache (direct store)
print("\nBenchmarking reads without cache...")
z_nocache = zarr.open('test3.zarr', mode='r')  # Direct access without cache
start = time.time()
for i in range(50):
    print(f'Reading iteration {i+1}/50 without cache...')
    _ = z_nocache[:]  # Read entire array
elapsed_nocache = time.time() - start

print(f"\n=== Results ===")
print(f"Read time with CacheStore: {elapsed_cache:.4f} s")
print(f"Read time without cache: {elapsed_nocache:.4f} s")
if elapsed_cache > 0:
    print(f"Speedup: {elapsed_nocache/elapsed_cache:.2f}x")

# Test cache effectiveness with specific chunks (like test2.py pattern)
print("\n=== Cache Effectiveness Test ===")
print('Testing first chunk access pattern...')

print('First access to chunk [0:10, 0:10] (should cache)')
start = time.time()
result1 = z_cached[0:10, 0:10]
first_access = time.time() - start

print('Second access to same chunk [0:10, 0:10] (should hit cache)')
start = time.time()
result2 = z_cached[0:10, 0:10]
second_access = time.time() - start

print('Third access to same chunk [0:10, 0:10] (should hit cache again)')
start = time.time()
result3 = z_cached[0:10, 0:10]
third_access = time.time() - start

print(f"\nChunk access timing:")
print(f"First access time: {first_access:.6f} s")
print(f"Second access time: {second_access:.6f} s")
print(f"Third access time: {third_access:.6f} s")

if second_access > 0:
    print(f"Cache speedup (1st vs 2nd): {first_access/second_access:.2f}x")
if third_access > 0:
    print(f"Cache speedup (1st vs 3rd): {first_access/third_access:.2f}x")

# Verify data consistency
print(f"\nData consistency check:")
print(f"All three reads identical: {np.array_equal(result1, result2) and np.array_equal(result2, result3)}")

# Test cache behavior with different chunks
print("\n=== Different Chunk Access Test ===")
print('Accessing different chunk [20:30, 20:30] (new cache entry)')
start = time.time()
result4 = z_cached[20:30, 20:30]
new_chunk_first = time.time() - start

print('Second access to new chunk [20:30, 20:30] (should hit cache)')
start = time.time()
result5 = z_cached[20:30, 20:30]
new_chunk_second = time.time() - start

print(f"New chunk first access: {new_chunk_first:.6f} s")
print(f"New chunk second access: {new_chunk_second:.6f} s")
if new_chunk_second > 0:
    print(f"New chunk cache speedup: {new_chunk_first/new_chunk_second:.2f}x")

print(f"Different chunk data identical: {np.array_equal(result4, result5)}")

# Test cache expiration simulation
print("\n=== Cache Behavior Summary ===")
print(f"Total operations completed successfully")
print(f"Cache store type: {type(cached_store._cache).__name__}")
print(f"Source store type: {type(cached_store._store).__name__}")
print(f"Max age setting: {cached_store.max_age_seconds} seconds")

print(f"\n=== Test Complete ===")
