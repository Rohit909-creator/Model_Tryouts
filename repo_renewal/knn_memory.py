import os
import math
import torch
import faiss
import numpy as np
from pathlib import Path
from functools import wraps
from typing import Optional, List, Tuple, Union

from contextlib import ExitStack, contextmanager

try:
    from einops import rearrange, pack, unpack
except ImportError:
    raise ImportError("Please install einops: pip install einops")

# multiprocessing
try:
    from joblib import Parallel, delayed, cpu_count
except ImportError:
    # Fallback without joblib
    import multiprocessing
    def cpu_count():
        return multiprocessing.cpu_count()
    
    def delayed(func):
        return func
    
    class Parallel:
        def __init__(self, n_jobs=1):
            self.n_jobs = n_jobs
        
        def __call__(self, tasks):
            return [task() if callable(task) else task for task in tasks]

# constants
FAISS_INDEX_GPU_ID = int(os.getenv('FAISS_INDEX_GPU_ID', 0))
DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY = './.tmp/knn.memories'

# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_list(val):
    return val if isinstance(val, list) else [val]

def all_el_unique(arr):
    return len(set(arr)) == len(arr)

@contextmanager
def multi_context(*cms):
    with ExitStack() as stack:
        yield [stack.enter_context(cls) for cls in cms]

def count_intersect(x, y):
    """Returns an array that shows how many times an element in x is contained in tensor y"""
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(rearrange(x, 'i -> i 1') == rearrange(y, 'j -> 1 j'), axis=-1)

def check_shape(tensor, pattern, **kwargs):
    return rearrange(tensor, f"{pattern} -> {pattern}", **kwargs)

# Modern FAISS wrapper that works with current installations
class KNN:
    def __init__(
        self,
        dim: int,
        max_num_entries: int,
        cap_num_entries: bool = False,
        M: int = 15,
        keep_stats: bool = False,
        use_gpu: bool = False
    ):
        """
        Initialize KNN index with HNSW (Hierarchical Navigable Small World) algorithm
        
        Args:
            dim: Dimension of vectors
            max_num_entries: Maximum number of entries
            cap_num_entries: Whether to cap entries at max_num_entries
            M: HNSW parameter (number of connections)
            keep_stats: Whether to keep statistics
            use_gpu: Whether to try using GPU (falls back to CPU if unavailable)
        """
        self.dim = dim
        self.max_num_entries = max_num_entries
        self.cap_num_entries = cap_num_entries
        self.keep_stats = keep_stats
        self.use_gpu = use_gpu
        
        # Create FAISS index - HNSW is more stable than IVF
        cpu_index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        
        # Try to move to GPU if requested and available
        if use_gpu and faiss.get_num_gpus() > 0:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, FAISS_INDEX_GPU_ID, cpu_index)
                self.is_gpu = True
            except Exception as e:
                print(f"Warning: Could not use GPU for FAISS, falling back to CPU: {e}")
                self.index = cpu_index
                self.is_gpu = False
        else:
            self.index = cpu_index
            self.is_gpu = False
        
        self.is_trained = False
        self.reset()

    def __del__(self):
        if hasattr(self, 'index'):
            del self.index

    def reset(self):
        """Reset the index and statistics"""
        self.ids = np.empty((0,), dtype=np.int64)  # Use int64 for better compatibility

        if self.keep_stats:
            self.hits = np.empty((0,), dtype=np.int32)
            self.age_num_iterations = np.empty((0,), dtype=np.int32)
            self.ages_since_last_hit = np.empty((0,), dtype=np.int32)

        self.index.reset()
        self.is_trained = False

    def train(self, x: np.ndarray):
        """Train the index if needed"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        
        # Ensure contiguous array
        x = np.ascontiguousarray(x.astype(np.float32))
        
        # HNSW doesn't need explicit training, but we call it for consistency
        if hasattr(self.index, 'train'):
            self.index.train(x)
        self.is_trained = True

    def add(self, x: np.ndarray, ids: np.ndarray):
        """Add vectors to the index"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        if not isinstance(ids, np.ndarray):
            ids = np.asarray(ids, dtype=np.int64)
        
        # Ensure proper data types and contiguous arrays
        x = np.ascontiguousarray(x.astype(np.float32))
        ids = np.ascontiguousarray(ids.astype(np.int64))
        
        if not self.is_trained:
            self.train(x)

        self.ids = np.concatenate((ids, self.ids))

        if self.keep_stats:
            self.hits = np.concatenate((np.zeros_like(ids, dtype=np.int32), self.hits))
            self.age_num_iterations = np.concatenate((np.zeros_like(ids, dtype=np.int32), self.age_num_iterations))
            self.ages_since_last_hit = np.concatenate((np.zeros_like(ids, dtype=np.int32), self.ages_since_last_hit))

        if self.cap_num_entries and len(self.ids) > self.max_num_entries:
            self.reset()
            return self.add(x, ids)  # Re-add after reset

        return self.index.add(x)

    def search(
        self,
        x: np.ndarray,
        topk: int,
        nprobe: int = 8,
        return_distances: bool = False,
        increment_hits: bool = False,
        increment_age: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Search for nearest neighbors"""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        
        x = np.ascontiguousarray(x.astype(np.float32))
        
        if not self.is_trained or self.index.ntotal == 0:
            # Return empty results if not trained or no data
            batch_size = x.shape[0] if x.ndim > 1 else 1
            empty_indices = np.full((batch_size, topk), -1, dtype=np.int64)
            if return_distances:
                empty_distances = np.full((batch_size, topk), float('inf'), dtype=np.float32)
                return empty_indices, empty_distances
            return empty_indices

        # Ensure we don't search for more items than exist
        actual_topk = min(topk, self.index.ntotal)
        
        try:
            distances, indices = self.index.search(x, k=actual_topk)
            
            # Pad results if we found fewer than requested
            if actual_topk < topk:
                batch_size = distances.shape[0]
                padded_distances = np.full((batch_size, topk), float('inf'), dtype=np.float32)
                padded_indices = np.full((batch_size, topk), -1, dtype=np.int64)
                
                padded_distances[:, :actual_topk] = distances
                padded_indices[:, :actual_topk] = indices
                
                distances = padded_distances
                indices = padded_indices
            
        except Exception as e:
            print(f"Warning: FAISS search failed: {e}")
            batch_size = x.shape[0] if x.ndim > 1 else 1
            indices = np.full((batch_size, topk), -1, dtype=np.int64)
            distances = np.full((batch_size, topk), float('inf'), dtype=np.float32)

        if increment_hits and self.keep_stats and len(self.ids) > 0:
            try:
                valid_indices = indices[indices != -1]
                if len(valid_indices) > 0:
                    hits = count_intersect(self.ids, valid_indices.flatten())
                    self.hits += hits

                    self.ages_since_last_hit += 1
                    self.ages_since_last_hit *= (hits == 0)
            except Exception as e:
                print(f"Warning: Could not update hit statistics: {e}")

        if increment_age and self.keep_stats:
            self.age_num_iterations += 1

        if return_distances:
            return indices, distances

        return indices

# KNN memory layer with improved error handling and modern practices
class KNNMemory:
    def __init__(
        self,
        dim: int,
        max_memories: int = 16000,
        num_indices: int = 1,
        memmap_filename: str = './knn.memory.memmap',
        multiprocessing: bool = True,
        use_gpu: bool = False
    ):
        self.dim = dim
        self.num_indices = num_indices
        self.scoped_indices = list(range(num_indices))
        self.max_memories = max_memories
        self.use_gpu = use_gpu
        
        self.shape = (num_indices, max_memories, 2, dim)
        self.db_offsets = np.zeros(num_indices, dtype=np.int64)

        # Create memmap file with better error handling
        try:
            memmap_path = Path(memmap_filename)
            memmap_path.parent.mkdir(parents=True, exist_ok=True)
            self.db = np.memmap(str(memmap_path), mode='w+', dtype=np.float32, shape=self.shape)
        except Exception as e:
            print(f"Warning: Could not create memmap file {memmap_filename}: {e}")
            print("Falling back to regular numpy array (will use more RAM)")
            self.db = np.zeros(self.shape, dtype=np.float32)
        
        # Initialize KNN indices
        self.knns = [
            KNN(
                dim=dim, 
                max_num_entries=max_memories, 
                cap_num_entries=True,
                use_gpu=use_gpu
            ) 
            for _ in range(num_indices)
        ]
    
        self.n_jobs = cpu_count() if multiprocessing else 1

    def set_scoped_indices(self, indices: List[int]):
        """Set which batch indices to operate on"""
        indices = list(indices)
        assert all_el_unique(indices), f'all scoped batch indices must be unique, received: {indices}'
        assert all([0 <= i < self.num_indices for i in indices]), f'each batch index must be between 0 and less than {self.num_indices}: received {indices}'
        self.scoped_indices = indices

    @contextmanager
    def at_batch_indices(self, indices: List[int]):
        """Context manager to temporarily change scoped indices"""
        prev_indices = self.scoped_indices
        self.set_scoped_indices(indices)
        try:
            yield self
        finally:
            self.set_scoped_indices(prev_indices)

    def clear(self, batch_indices: Optional[List[int]] = None):
        """Clear memory for specified batch indices"""
        if not exists(batch_indices):
            batch_indices = list(range(self.num_indices))

        batch_indices = cast_list(batch_indices)

        for index in batch_indices:
            if 0 <= index < len(self.knns):
                knn = self.knns[index]
                knn.reset()

        self.db_offsets[batch_indices] = 0

    def add(self, memories: torch.Tensor):
        """Add memories to the KNN indices"""
        check_shape(memories, 'b n kv d', d=self.dim, kv=2, b=len(self.scoped_indices))

        memories = memories.detach().cpu().numpy()
        memories = memories[:, -self.max_memories:]
        num_memories = memories.shape[1]

        if num_memories == 0:
            return

        knn_insert_ids = np.arange(num_memories, dtype=np.int64)
        keys = np.ascontiguousarray(memories[..., 0, :])
        knns = [self.knns[i] for i in self.scoped_indices]
        db_offsets = [self.db_offsets[i] for i in self.scoped_indices]

        # Use joblib to insert new key / value memories into faiss index
        if hasattr(delayed, '__call__'):  # Check if joblib is available
            @delayed
            def knn_add(knn, key, db_offset):
                knn.add(key, ids=knn_insert_ids + db_offset)
                return knn

            try:
                updated_knns = Parallel(n_jobs=self.n_jobs)(
                    knn_add(*args) for args in zip(knns, keys, db_offsets)
                )
                for knn_idx, scoped_idx in enumerate(self.scoped_indices):
                    self.knns[scoped_idx] = updated_knns[knn_idx]
            except Exception as e:
                print(f"Warning: Parallel processing failed, falling back to sequential: {e}")
                # Fallback to sequential processing
                for knn_idx, (knn, key, db_offset) in enumerate(zip(knns, keys, db_offsets)):
                    knn.add(key, ids=knn_insert_ids + db_offset)
        else:
            # Sequential processing when joblib is not available
            for knn_idx, (knn, key, db_offset) in enumerate(zip(knns, keys, db_offsets)):
                knn.add(key, ids=knn_insert_ids + db_offset)

        # Add the new memories to the memmap "database"
        try:
            add_indices = (
                rearrange(np.arange(num_memories), 'j -> 1 j') + 
                rearrange(self.db_offsets[list(self.scoped_indices)], 'i -> i 1')
            ) % self.max_memories
            
            self.db[rearrange(np.array(self.scoped_indices), 'i -> i 1'), add_indices] = memories
            
            # Flush if it's a memmap
            if hasattr(self.db, 'flush'):
                self.db.flush()
        except Exception as e:
            print(f"Warning: Could not update memory database: {e}")

        self.db_offsets[list(self.scoped_indices)] += num_memories

    def search(
        self,
        queries: torch.Tensor,
        topk: int,
        nprobe: int = 8,
        increment_hits: bool = True,
        increment_age: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Search for memories similar to queries"""
        check_shape(queries, 'b ... d', d=self.dim, b=len(self.scoped_indices))
        queries, ps = pack([queries], 'b * d')

        device = queries.device
        queries_np = queries.detach().cpu().numpy()

        all_masks = []
        all_key_values = []

        knns = [self.knns[i] for i in self.scoped_indices]

        # Parallelize faiss search if joblib is available
        if hasattr(delayed, '__call__'):
            @delayed
            def knn_search(knn, query):
                return knn.search(
                    query, topk, nprobe, 
                    increment_hits=increment_hits, 
                    increment_age=increment_age
                )

            try:
                fetched_indices = Parallel(n_jobs=self.n_jobs)(
                    knn_search(*args) for args in zip(knns, queries_np)
                )
            except Exception as e:
                print(f"Warning: Parallel search failed, using sequential: {e}")
                fetched_indices = [
                    knn.search(query, topk, nprobe, increment_hits=increment_hits, increment_age=increment_age)
                    for knn, query in zip(knns, queries_np)
                ]
        else:
            # Sequential search when joblib is not available
            fetched_indices = [
                knn.search(query, topk, nprobe, increment_hits=increment_hits, increment_age=increment_age)
                for knn, query in zip(knns, queries_np)
            ]

        # Get all the memory key / values from database
        for batch_index, indices in zip(self.scoped_indices, fetched_indices):
            mask = indices != -1
            db_indices = np.where(mask, indices, 0)

            all_masks.append(torch.from_numpy(mask))

            try:
                key_values = self.db[batch_index, db_indices % self.max_memories]
                all_key_values.append(torch.from_numpy(key_values.copy()))
            except Exception as e:
                print(f"Warning: Could not retrieve key-values for batch {batch_index}: {e}")
                # Create empty key-values as fallback
                empty_kv = np.zeros((indices.shape[0], indices.shape[1], 2, self.dim), dtype=np.float32)
                all_key_values.append(torch.from_numpy(empty_kv))

        all_masks = torch.stack(all_masks)
        all_key_values = torch.stack(all_key_values)
        all_key_values = all_key_values.masked_fill(~rearrange(all_masks, '... -> ... 1 1'), 0.)

        all_key_values, = unpack(all_key_values, ps, 'b * n kv d')
        all_masks, = unpack(all_masks, ps, 'b * n')

        return all_key_values.to(device), all_masks.to(device)

    def __del__(self):
        if hasattr(self, 'knns'):
            for knn in self.knns:
                del knn
        if hasattr(self, 'db'):
            del self.db

# Extended list class for collections of KNN memories
class KNNMemoryList(list):
    def cleanup(self):
        """Clean up all memories in the list"""
        for memory in self:
            del memory
        self.clear()

    @classmethod
    def create_memories(
        cls,
        *,
        batch_size: int,
        num_memory_layers: int,
        memories_directory: str = DEFAULT_KNN_MEMORY_MEMMAP_DIRECTORY,
        use_gpu: bool = False
    ):
        """Factory method to create multiple KNN memories"""
        memories_path = Path(memories_directory)
        memories_path.mkdir(exist_ok=True, parents=True)

        def inner(*args, **kwargs):
            return cls([
                KNNMemory(
                    *args, 
                    num_indices=batch_size, 
                    memmap_filename=str(memories_path / f'knn.memory.layer.{ind + 1}.memmap'),
                    use_gpu=use_gpu,
                    **kwargs
                ) 
                for ind in range(num_memory_layers)
            ])
        return inner

    @contextmanager
    def at_batch_indices(self, indices: List[int]):
        """Context manager to operate on specific batch indices across all memories"""
        knn_batch_indices_contexts = [memory.at_batch_indices(indices) for memory in self]
        with multi_context(*knn_batch_indices_contexts):
            yield

    def clear_memory(
        self,
        batch_indices: Optional[List[int]] = None,
        memory_indices: Optional[List[int]] = None
    ):
        """Clear memory for specified batch and memory indices"""
        memory_indices = default(memory_indices, tuple(range(len(self))))

        for memory_index in memory_indices:
            if 0 <= memory_index < len(self):
                memory = self[memory_index]
                memory.clear(batch_indices)