import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dgl
import dgl.data.utils as U
import time
import pickle
from models.layers import *
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import copy
from sklearn.metrics import precision_score,f1_score,recall_score
import warnings
warnings.filterwarnings('ignore')

# PERFORMANCE OPTIMIZATIONS
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Latest cuDNN API
os.environ['OMP_NUM_THREADS'] = '8'  # Optimize CPU threads

import torch.backends.cudnn as cudnn

# Global PyTorch optimizations
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('medium')  # Use Tensor Cores
cudnn.benchmark = True  # Optimize for consistent input sizes
cudnn.deterministic = False  # Allow non-deterministic for speed

class UnircaDataset():
    """
    参数
    ----------
    dataset_path: str
        数据存放位置。
        举例: 'train_Xs.pkl' （67 * 14 * 40）（图数 * 节点数 * 节点向量维数）
    labels_path: str
        标签存放位置。
        举例: 'train_ys_anomaly_type.pkl' （67）
    topology: str
        图的拓扑结构存放位置
        举例：'topology.pkl'
    aug: boolean (default: False)
        需要数据增强，该值设置为True
    aug_size: int (default: 0)
        数据增强时，每个label对应的样本数
    shuffle: boolean (default: False)
        load()完成以后，若shuffle为True，则打乱self.graphs 和 self.labels （同步）
    """
    def __init__(self, dataset_path, labels_path, topology, aug=False, aug_size=0, shuffle=False):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.topology_path = topology
        self.aug = aug
        self.aug_size = aug_size
        self.shuffle_on_load = shuffle
        
        self.graphs = []
        self.labels = []
        self.chunk_files = []
        self._chunk_cache = {}
        self._max_cache_size = 5 # Number of chunks to keep in RAM
        
        self.load()
        if self.shuffle_on_load and not self.chunk_files:
             self.shuffle()

    def __len__(self):
        # Use the total size calculated from chunks, not just len(self.labels)
        if hasattr(self, '_total_size'):
            return self._total_size
        return len(self.labels)

    def load(self):
        """OPTIMIZED: Determines whether to load all data at once or prepare for streaming with memory management."""
        import gc
        
        # Clear any existing data to free memory
        if hasattr(self, 'graphs') and self.graphs:
            del self.graphs
        if hasattr(self, '_chunk_cache'):
            self._chunk_cache.clear()
        gc.collect()
        
        # If the path is a directory ending in '_chunked', use streaming mode.
        if os.path.isdir(self.dataset_path):
            print(f"📦 Path '{self.dataset_path}' is a directory. Initializing OPTIMIZED streaming mode.")
            try:
                self._initialize_streaming()
            except FileNotFoundError as e:
                 print(f"[ERROR] Streaming initialization failed: {e}")
                 # Optionally fallback or raise error
                 raise RuntimeError("Failed to initialize streaming mode. Check chunk files.") from e
            except Exception as e:
                 print(f"[ERROR] Unexpected error during streaming init: {e}")
                 raise
        # Otherwise, assume it's a single file for traditional mode.
        else:
            print(f"📁 Path '{self.dataset_path}' is not a directory. Loading all data into memory (traditional mode).")
            try:
                self._load_regular()
            except FileNotFoundError:
                 raise FileNotFoundError(f"Traditional mode failed: File not found at '{self.dataset_path}'")
            except Exception as e:
                 print(f"[ERROR] Unexpected error during traditional load: {e}")
                 raise
        # Force garbage collection after loading
        gc.collect()

    def _initialize_streaming(self):
        """OPTIMIZED: Prepares the dataset for streaming chunks from disk with performance enhancements."""
        import glob
        import bisect  # Import bisect

        self.chunk_files = sorted(glob.glob(os.path.join(self.dataset_path, "*.pkl")))
        if not self.chunk_files:
            raise FileNotFoundError(f"No chunk files (.pkl) found in directory: {self.dataset_path}")

        # Load the corresponding labels
        self.labels = tensor(U.load_info(self.labels_path))
        self.topology = U.load_info(self.topology_path)

        # --- START: CHUNK OFFSET CALCULATION ---
        # This is the robust fix for the IndexError
        print("🔧 Calculating chunk offsets (this may take a moment)...")
        self.chunk_offsets = [0]
        self.chunk_sizes = []
        total_items = 0

        # Load each chunk to find its size. This is slow but robust.
        # A better way would be to load a metadata file if it exists.
        for chunk_file_path in self.chunk_files:
            try:
                with open(chunk_file_path, 'rb') as f:
                    chunk_data = pickle.load(f)
                    chunk_len = len(chunk_data)
                    self.chunk_sizes.append(chunk_len)
                    total_items += chunk_len
                    self.chunk_offsets.append(total_items)
            except Exception as e:
                print(f"⚠️ Failed to load chunk {chunk_file_path} for size check: {e}")
                raise
        
        # self.chunk_offsets will be [0, len(c0), len(c0)+len(c1), ...]
        # We pop the last item so it's a list of *start* indices
        self.chunk_offsets.pop()
        self._total_size = total_items
        
        # This is the old self.chunk_size, now we just need it for one check
        self.chunk_size = self.chunk_sizes[0]
        print(f"✅ Chunk offsets calculated: {self._total_size} total items across {len(self.chunk_files)} chunks.")

        if len(self.labels) != self._total_size:
            print(f"⚠️ [WARNING] Label count ({len(self.labels)}) does not match total items in chunks ({self._total_size})!")
            # Trust the chunks, as they are the data source.
            if len(self.labels) > self._total_size:
                self.labels = self.labels[:self._total_size] # Truncate labels
            # Note: Padding labels if too short is complex, assuming truncation is the main case.
            
        # --- END: CHUNK OFFSET CALCULATION ---

        # Create placeholders for graphs. They will be loaded lazily.
        self.graphs = [None] * self._total_size  # Use new total size

        # OPTIMIZED Cache management - larger cache for better performance
        from collections import OrderedDict
        self._chunk_cache = OrderedDict()  # LRU-style cache

        # Get cache size from config (default optimized to 20)
        config_cache_size = getattr(self, 'config', {}).get('max_cache_size', 20)
        self._max_cache_size = max(5, config_cache_size)  # Minimum 5 chunks

        print(f" OPTIMIZED Cache: {self._max_cache_size} chunks (LRU eviction)")

        # Pre-compile DGL graph template for faster graph creation
        self._graph_template = dgl.graph(self.topology)
        in_degrees = self._graph_template.in_degrees()
        zero_indegree_nodes = [i for i, deg in enumerate(in_degrees) if deg == 0]
        if zero_indegree_nodes:
            self._graph_template.add_edges(zero_indegree_nodes, zero_indegree_nodes)

        print(f"🔧 Pre-compiled graph template: {self._graph_template.number_of_nodes()} nodes, {self._graph_template.number_of_edges()} edges")

        # CRITICAL: Load the first graph immediately for training dimension detection
        if self._total_size > 0 and self.chunk_sizes[0] > 0:
            print("🔧 Loading first graph for dimension detection...")
            
            # Need to re-load chunk 0 into cache (it was loaded for size check but not cached)
            with open(self.chunk_files[0], 'rb') as f:
                first_chunk = pickle.load(f)

            X = tensor(first_chunk[0])
            g = self._graph_template.clone()  # Fast clone from template
            g.ndata['attr'] = X
            self.graphs[0] = g

            # Cache the first chunk with LRU management
            self._chunk_cache[0] = first_chunk

            print(f"✅ First graph loaded: {X.shape[1]} features, {g.number_of_nodes()} nodes")

        # Pre-load additional chunks for better cache hit rate
        num_preload = min(3, len(self.chunk_files) - 1, self._max_cache_size - 1)
        if num_preload > 0:
            print(f"🚀 Pre-loading {num_preload} additional chunks for performance...")
            for i in range(1, num_preload + 1):
                with open(self.chunk_files[i], 'rb') as f:
                    self._chunk_cache[i] = pickle.load(f)

        print(f"✅ OPTIMIZED Streaming ready: {self._total_size} graphs, {len(self.chunk_files)} chunks, {len(self._chunk_cache)} pre-cached")
        
    def _load_chunk(self, chunk_idx):
        """OPTIMIZED: Loads a specific chunk into the cache with LRU management and performance enhancements."""
        # Check if already cached (move to end for LRU)
        if chunk_idx in self._chunk_cache:
            # Move to end (most recently used)
            self._chunk_cache.move_to_end(chunk_idx)
            return
        
        # LRU eviction: remove least recently used chunk if cache is full
        if len(self._chunk_cache) >= self._max_cache_size:
            oldest_key, _ = self._chunk_cache.popitem(last=False)  # Remove least recent
        
        # Bounds check
        if chunk_idx >= len(self.chunk_files):
            raise IndexError(f"Chunk index {chunk_idx} out of range (max: {len(self.chunk_files)-1})")
        
        # OPTIMIZED: Load the new chunk with buffered I/O
        chunk_file_path = self.chunk_files[chunk_idx]
        try:
            with open(chunk_file_path, 'rb', buffering=8192) as f:  # 8KB buffer
                chunk_data = pickle.load(f)
                self._chunk_cache[chunk_idx] = chunk_data
        except Exception as e:
            print(f"⚠️ Failed to load chunk {chunk_idx} from {chunk_file_path}: {e}")
            raise

    def __getitem__(self, idx):
        """
        OPTIMIZED: Provides a single data item (graph, label) with performance enhancements.
        Uses graph template cloning and optimized tensor operations.
        Handles data shape mismatches by padding.
        Uses chunk_offsets for robust irregular chunk indexing.
        """
        import bisect
        
        # Check if we're in streaming mode (chunk_files exist)
        if hasattr(self, 'chunk_files') and self.chunk_files:
            # If graph hasn't been loaded yet, load it on-demand
            if self.graphs[idx] is None:
                
                # --- START: ROBUST CHUNK INDEXING ---
                if not hasattr(self, 'chunk_offsets'):
                     # Fallback or error if streaming wasn't initialized correctly
                     raise RuntimeError("chunk_offsets not found. Please re-run initialization.")
                
                # Find the correct chunk_idx using binary search on start-offsets
                chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
                
                # Find the index *within* that chunk
                case_in_chunk_idx = idx - self.chunk_offsets[chunk_idx]
                # --- END: ROBUST CHUNK INDEXING ---
                
                # Load the required chunk if it's not already in our cache
                if chunk_idx not in self._chunk_cache:
                    self._load_chunk(chunk_idx) # This should be safe now
                
                # Get the specific case from the cached chunk
                chunk_data = self._chunk_cache[chunk_idx]
                
                # This check should be more robust now, but we keep it
                if case_in_chunk_idx >= len(chunk_data):
                    print(f"⚠️ [WARNING] case_in_chunk_idx ({case_in_chunk_idx}) out of bounds for chunk {chunk_idx} (len {len(chunk_data)}). Clamping.")
                    case_in_chunk_idx = len(chunk_data) - 1
                
                # OPTIMIZED: Get the feature data with efficient tensor conversion
                try:
                    X = tensor(chunk_data[case_in_chunk_idx], dtype=torch.float32)
                except Exception as e:
                    print(f"⚠️ Failed to convert data at idx {idx} to tensor: {e}")
                    X = tensor(chunk_data[case_in_chunk_idx])
                
                # OPTIMIZED: Build the DGL graph
                if hasattr(self, '_graph_template'):
                    g = self._graph_template.clone()
                else:
                    g = dgl.graph(self.topology)
                    in_degrees = g.in_degrees()
                    zero_indegree_nodes = [i for i, deg in enumerate(in_degrees) if deg == 0]
                    if zero_indegree_nodes:
                        g.add_edges(zero_indegree_nodes, zero_indegree_nodes)
                
                # --- START: ROBUST FIX FOR SHAPE MISMATCH (No prints) ---
                expected_nodes = g.number_of_nodes()
                current_rows = X.shape[0]

                if current_rows != expected_nodes:
                    # This print statement was REMOVED
                    # print(f"⚠️  [FIX] Data mismatch at index {idx}: Got {current_rows} rows, expected {expected_nodes} nodes.")
                    
                    if self.graphs[0] is None or 'attr' not in self.graphs[0].ndata:
                        raise RuntimeError(f"Dataset not initialized correctly. self.graphs[0] is missing. Failed at index {idx}.")
                    
                    feature_dim = self.graphs[0].ndata['attr'].shape[1] 
                    
                    padded_X = torch.zeros((expected_nodes, feature_dim), dtype=X.dtype, device=X.device)
                    
                    if current_rows > 0 and X.dim() == 2:
                        padded_X[:current_rows, :] = X
                    
                    X = padded_X
                    # This print statement was REMOVED
                    # print(f"    Successfully padded to {X.shape}.")
                # --- END: ROBUST FIX FOR SHAPE MISMATCH ---

                g.ndata['attr'] = X
                self.graphs[idx] = g

        # Handle label indexing, ensuring it doesn't go out of bounds
        label_idx = min(idx, len(self.labels) - 1)
        if label_idx < 0: label_idx = 0 # Safety for empty labels
        
        return self.graphs[idx], self.labels[label_idx]
    
    def _load_regular(self):
        """The original method to load a single large .pkl file."""
        # This part remains the same as your original code
        try:
            import public_function as pf
            Xs = pf.load_chunked(self.dataset_path)
        except Exception:
            Xs = U.load_info(self.dataset_path)
            
        Xs = tensor(Xs)
        ys = tensor(U.load_info(self.labels_path))
        topology = U.load_info(self.topology_path)
        
        for X in Xs:
            g = dgl.graph(topology)
            g.ndata['attr'] = X
            self.graphs.append(g)
        self.labels = ys

    def shuffle(self):
        # Note: True shuffling in streaming mode is complex. This shuffles labels
        # and will cause chunks to be loaded in a random order, which is good enough.
        combined = list(zip(self.graphs, self.labels))
        random.shuffle(combined)
        self.graphs[:], self.labels[:] = zip(*combined)

    def aug_data(self, Xs, ys):
        """ load() 中使用，作用是数据增强
        参数
        ----------
        Xs: tensor
            多个图对应的特征向量矩阵。
            举例：67个图对应的Xs规模为 67 * 14 * 40 （67个图，每个图14个节点）
        ys: tensor
            每个图对应的label，要求是从0开始的整数。
            举例：如果一共有10个label，那么ys中元素值为 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        self.aug_size: int
            数据增强时，每个label对应的样本数

        返回值
        ----------
        aug_Xs: tensor
            数据增强的结果
        aug_ys: tensor
            数据增强的结果
        """
        aug_Xs = []
        aug_ys = []
        num_label = len(set([y.item() for y in ys]))
        grouped_Xs = [[] for i in range(num_label)]
        for X, y in zip(Xs, ys):
            grouped_Xs[y.item()].append(X)
        for group_idx in range(len(grouped_Xs)):
            cur_Xs = grouped_Xs[group_idx]
            n = len(cur_Xs)
            m = Xs.shape[1]
            while len(cur_Xs) < self.aug_size:
                select = np.random.choice(n, m)
                aug_X = torch.zeros_like(Xs[0])
                for i, j in zip(select, range(m)):
                    aug_X[j] = cur_Xs[i][j].detach().clone()
                cur_Xs.append(aug_X)
            for X in cur_Xs:
                aug_Xs.append(X)
                aug_ys.append(group_idx)
        aug_Xs = torch.stack(aug_Xs, 0)
        aug_ys = tensor(aug_ys)
        return aug_Xs, aug_ys


# ...existing code...

class RawDataProcess():
    """用来处理原始数据的类
    参数
    ----------
    config: dict
        配置参数
        Xs: 多个图的特征向量矩阵
        data_dir: 数据和结果存放路径
        dataset: 数据集名称 可选['21aiops', 'gaia']
    """

    def __init__(self, config):
        self.config = config

    # Replace the process() method in RawDataProcess class:
    # In He_DGL.py
# --- Replace the process method in the RawDataProcess class ---

    # Inside the RawDataProcess class in He_DGL.py
    # Inside the RawDataProcess class in He_DGL.py
    def process(self):
        """
        Prepares label files and correctly handles traditional vs. streaming embeddings.
        Robust: discovers metadata/chunk folder variants, validates paths, and writes
        streaming placeholders into the processed-data subfolder(s) where the loader expects them.
        """
        import pickle, gc

        # --- Load Run Table ---
        run_table_path = os.path.join(self.config['data_dir'], self.config['run_table'])
        try:
            run_table = pd.read_csv(run_table_path)
            if 'data_type' not in run_table.columns:
                print("[WARNING] 'data_type' column missing in run_table. Attempting 80/20 split based on index.")
                train_size = int(len(run_table) * 0.8)
                run_table['data_type'] = ['train'] * train_size + ['test'] * (len(run_table) - train_size)
        except FileNotFoundError:
            raise FileNotFoundError(f"Run table file not found at: {run_table_path}")
        except ValueError as e:
            raise ValueError(f"Error processing run_table columns: {e}")

        save_dir = self.config['save_dir']
        os.makedirs(save_dir, exist_ok=True) # Ensure save_dir exists

        # --- Process and Save Labels ---
        print("📊 Processing and splitting label files...")
        label_types = ['anomaly_type', 'service']
        label_dict = {}

        # --- CORRECTED INDEX SPLITTING WITH FALLBACK ---
        train_index = np.array([], dtype=int) # Initialize as empty
        test_index = np.array([], dtype=int)  # Initialize as empty

        if 'data_type' in run_table.columns and \
           run_table['data_type'].isin(['train', 'test']).any():
            print("[INFO] Attempting split based on 'data_type' column.")
            train_index = np.where(run_table['data_type'].values == 'train')[0]
            test_index = np.where(run_table['data_type'].values == 'test')[0]

        # Check if indices were successfully found using 'data_type'
        if len(train_index) == 0 or len(test_index) == 0:
            if 'data_type' not in run_table.columns:
                 print("[WARNING] 'data_type' column missing in run_table.")
            else:
                 print("[WARNING] No 'train' or 'test' values found (or usable) in 'data_type' column.")
            print("[INFO] Using fallback 80/20 split based on index.")
            num_samples = len(run_table)
            if num_samples == 0:
                 print("[ERROR] Run table is empty! Cannot split indices.")
                 # Handle appropriately - maybe raise error or continue with empty indices
            else:
                train_size = int(num_samples * 0.8)
                all_indices = np.arange(num_samples)
                # np.random.shuffle(all_indices) # Optional: Shuffle before splitting
                train_index = all_indices[:train_size]
                test_index = all_indices[train_size:]

        # Final check if indices were successfully created by either method
        if len(train_index) == 0 or len(test_index) == 0:
             # This should only happen if run_table was empty initially
             print("[CRITICAL WARNING] Train or test index arrays are EMPTY. Label files will be empty.")
        else:
             print(f"   Train indices count: {len(train_index)}")
             print(f"   Test indices count: {len(test_index)}")
        # --- END INDEX SPLITTING CORRECTION ---

        # --- Label Saving Loop (Keep the detailed debugging from previous step) ---
        for label_type in label_types:
            print(f"\n--- Processing Label Type: {label_type} ---")
            try:
                 # Step 1: Get all labels for this type
                 labels = self.get_label(label_type, run_table)
                 print(f"   DEBUG: Got full labels array. Shape: {labels.shape}, Type: {labels.dtype}, Unique values: {np.unique(labels).tolist()}")

                 # Step 2: Slice the labels using the calculated indices
                 train_labels_to_save = np.array([]) # Default to empty
                 test_labels_to_save = np.array([])  # Default to empty

                 # Only slice if indices and labels are valid
                 if len(labels) > 0 and len(train_index) > 0 and train_index.max() < len(labels):
                      train_labels_to_save = labels[train_index]
                 elif len(train_index) > 0:
                      print(f"   [ERROR] Cannot slice train labels: Max train index ({train_index.max()}) >= label length ({len(labels)}) or labels empty.")

                 if len(labels) > 0 and len(test_index) > 0 and test_index.max() < len(labels):
                      test_labels_to_save = labels[test_index]
                 elif len(test_index) > 0:
                      print(f"   [ERROR] Cannot slice test labels: Max test index ({test_index.max()}) >= label length ({len(labels)}) or labels empty.")

                 print(f"   DEBUG: Sliced train labels shape: {train_labels_to_save.shape}")
                 print(f"   DEBUG: Sliced test labels shape: {test_labels_to_save.shape}")

                 # Step 3: Save the sliced labels
                 train_save_path = os.path.join(save_dir, f'train_ys_{label_type}.pkl')
                 test_save_path = os.path.join(save_dir, f'test_ys_{label_type}.pkl')

                 print(f"   DEBUG: Saving train labels to: {train_save_path}")
                 U.save_info(train_save_path, train_labels_to_save)
                 print(f"   DEBUG: Train labels saved? Exists: {os.path.exists(train_save_path)}, Size: {os.path.getsize(train_save_path) if os.path.exists(train_save_path) else 'N/A'} bytes")

                 print(f"   DEBUG: Saving test labels to: {test_save_path}")
                 U.save_info(test_save_path, test_labels_to_save)
                 print(f"   DEBUG: Test labels saved? Exists: {os.path.exists(test_save_path)}, Size: {os.path.getsize(test_save_path) if os.path.exists(test_save_path) else 'N/A'} bytes")

            except Exception as e_gen: # Catch other potential errors during label processing/saving
                 print(f"[ERROR] Unexpected error processing/saving labels for '{label_type}': {e_gen}")
                 raise # Re-raise

        print("✅ Label files processing complete.")

        # --- MODE DETECTION & HANDLING ---
        xs_base_path = self.config['Xs']  # e.g., ".../anomalies/sentence_embedding"
        base_dir = os.path.dirname(xs_base_path)
        base_name = os.path.basename(xs_base_path)

        # Candidate locations for metadata (parent dir and inside chunk folders)
        candidates_meta = [
            os.path.normpath(os.path.join(base_dir, f"{base_name}_metadata.pkl")),
            os.path.normpath(os.path.join(base_dir, f"{base_name}", f"{base_name}_metadata.pkl")),
            os.path.normpath(os.path.join(base_dir, f"{base_name}_chunked", f"{base_name}_metadata.pkl")),
            os.path.normpath(os.path.join(base_dir, f"{base_name}_chunks", f"{base_name}_metadata.pkl")),
        ]
        metadata_path = next((p for p in candidates_meta if os.path.exists(p)), candidates_meta[0])

        if os.path.exists(metadata_path):
            # STREAMING MODE
            print(f"📦 Streaming mode detected based on metadata file: {metadata_path}")
            print("   Skipping embedding splitting. UnircaDataset will handle streaming.")

            # Try likely chunk-folder names and pick the one that exists
            candidate_chunk_dirs = [
                os.path.normpath(f"{xs_base_path}_chunked"),
                os.path.normpath(f"{xs_base_path}_chunks"),
                os.path.normpath(os.path.join(base_dir, base_name, "sentence_embedding_chunked")),
                os.path.normpath(os.path.join(base_dir, base_name, "sentence_embedding_chunks")),
            ]
            chunked_dir = next((d for d in candidate_chunk_dirs if os.path.isdir(d)), None)
            if chunked_dir is None:
                raise FileNotFoundError(
                    f"Could not find any chunked directory for embeddings. Checked: {candidate_chunk_dirs}"
                )

            # Use absolute normalized chunk dir
            chunked_dir = os.path.abspath(chunked_dir)
            streaming_info = {'chunked_dir': chunked_dir}

            # Determine target folders where the loader will look for placeholders.
            # The loader earlier tried something like: <save_dir>/<lab_id>/train_Xs.pkl
            # So write placeholder into save_dir and into any numeric child directories (safest).
            target_dirs = [os.path.abspath(save_dir)]
            # add any immediate numeric subfolders (e.g., 9, 10) where loader might look
            try:
                for entry in os.listdir(save_dir):
                    full = os.path.join(save_dir, entry)
                    if os.path.isdir(full) and entry.isdigit():
                        target_dirs.append(os.path.abspath(full))
            except FileNotFoundError:
                pass

            # Also, if config explicitly provides an experiment id / subfolder, respect it
            exp_subdir = self.config.get('exp_subdir') or self.config.get('lab_id') or self.config.get('run_id')
            if exp_subdir:
                candidate = os.path.join(save_dir, str(exp_subdir))
                if os.path.isdir(candidate) and os.path.abspath(candidate) not in target_dirs:
                    target_dirs.append(os.path.abspath(candidate))

            # write placeholders in all target dirs (redundant but robust)
            for td in set(target_dirs):
                os.makedirs(td, exist_ok=True)
                placeholder_train = os.path.join(td, 'train_Xs_streaming.pkl')
                placeholder_test = os.path.join(td, 'test_Xs_streaming.pkl')
                try:
                    with open(placeholder_train, 'wb') as f:
                        pickle.dump(streaming_info, f, protocol=pickle.HIGHEST_PROTOCOL)
                    with open(placeholder_test, 'wb') as f:
                        pickle.dump(streaming_info, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"   ✅ Streaming placeholders created in: {td} -> chunk dir: {chunked_dir}")
                    # ... (saving placeholder_train) ...
                    print(f"DEBUG RawDataProcess SAVE: Path='{placeholder_train}', Exists={os.path.exists(placeholder_train)}") # ADD THIS

                    # ... (saving placeholder_test) ...
                    print(f"DEBUG RawDataProcess SAVE: Path='{placeholder_test}', Exists={os.path.exists(placeholder_test)}") # ADD THIS

                    print(f"✅ Streaming placeholders created successfully in: {save_dir}")
                except Exception as e:
                    print(f"[ERROR] Failed to save streaming placeholder files in {td}: {e}")
                    raise IOError(f"Could not create streaming placeholder files in {td}") from e

        else:
            # TRADITIONAL MODE
            print(f"📁 Metadata file not found at '{metadata_path}'.")
            single_xs_file = os.path.normpath(f"{xs_base_path}.pkl")
            print(f"   Assuming traditional mode with single file: {single_xs_file}")
            print("   Splitting embedding file (this may require significant memory)...")

            if not os.path.exists(single_xs_file):
                raise FileNotFoundError(f"Traditional mode failed: Input embedding file not found at {single_xs_file}")

            print(f"   Loading single embedding file: {single_xs_file}...")
            try:
                with open(single_xs_file, 'rb') as f:
                    Xs_all = pickle.load(f)
            except Exception as e:
                raise IOError(f"Failed to load single embedding file {single_xs_file}: {e}")

            if len(Xs_all) != len(run_table):
                print(f"[WARNING] Length mismatch! Embeddings ({len(Xs_all)}) vs Run Table ({len(run_table)}). Split might be incorrect.")

            print(f"   Splitting into train ({len(train_index)} cases) and test ({len(test_index)} cases)...")
            try:
                train_Xs = [Xs_all[i] for i in train_index]
                test_Xs = [Xs_all[i] for i in test_index]
            except IndexError:
                raise IndexError("Index out of bounds during train/test split. Check run_table indices vs embedding length.")
            except TypeError:
                raise TypeError("Failed to index embeddings. Ensure Xs_all is a list or indexable sequence.")

            train_file_path = os.path.join(save_dir, 'train_Xs.pkl')
            test_file_path = os.path.join(save_dir, 'test_Xs.pkl')

            print(f"   Saving train embeddings to: {train_file_path}")
            if 'pf' not in locals() and 'pf' not in globals():
                import public_function as pf
            try:
                if hasattr(pf, 'save_chunked'):
                    pf.save_chunked(train_file_path, train_Xs)
                else:
                    with open(train_file_path, 'wb') as f:
                        pickle.dump(train_Xs, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[ERROR] Failed saving train_Xs: {e}")

            print(f"   Saving test embeddings to: {test_file_path}")
            try:
                if hasattr(pf, 'save_chunked'):
                    pf.save_chunked(test_file_path, test_Xs)
                else:
                    with open(test_file_path, 'wb') as f:
                        pickle.dump(test_Xs, f, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print(f"[ERROR] Failed saving test_Xs: {e}")

            del Xs_all, train_Xs, test_Xs
            gc.collect()
            print("✅ Traditional splitting complete.")

        # --- Save Topology (Always happens) ---
        print("💾 Saving topology...")
        topology = self.get_topology()
        U.save_info(os.path.join(save_dir, 'topology.pkl'), topology)
        if self.config.get('heterogeneous', False):
            print("💾 Saving edge types...")
            edge_types = self.get_edge_types()
            U.save_info(os.path.join(save_dir, 'edge_types.pkl'), edge_types)
        print("✅ Topology/Edge types saved.")

                    
    def process_embeddings_chunked(self, xs_path, save_dir, train_index, test_index, label_dict):
        """Streaming process: never accumulate more than one chunk in memory"""
        import public_function as pf
        from tqdm import tqdm
        import pickle
        import os
        import psutil  # for memory monitoring

        try:
            base_dir = os.path.dirname(xs_path)
            base_filename = os.path.basename(xs_path).replace(".pkl", "")
            metadata_path = os.path.join(base_dir, f'{base_filename}_metadata.pkl')

            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            total_items = metadata['total_cases']
            chunk_manifest = metadata['chunks']
            
            print(f"✅ Streaming process: {total_items} cases from {len(chunk_manifest)} chunks")
            
            train_indices_set = set(train_index)
            
            # Initialize output files for streaming writes
            train_file = os.path.join(save_dir, 'train_Xs.pkl')
            test_file = os.path.join(save_dir, 'test_Xs.pkl')
            
            # Remove existing files
            for f in [train_file, test_file]:
                if os.path.exists(f):
                    os.remove(f)
            
            current_case_index = 0
            train_count = 0
            test_count = 0
            
            # Process one chunk at a time
            for chunk_idx, chunk_info in enumerate(tqdm(chunk_manifest, desc="Streaming chunks")):
                chunk_filename = os.path.basename(chunk_info['path'])
                chunked_dir = os.path.join(base_dir, 'sentence_embedding_chunked')
                chunk_path = os.path.join(chunked_dir, chunk_filename)
                
                if not os.path.exists(chunk_path):
                    current_case_index += 500  # Assume standard chunk size
                    continue

                # Progress monitoring
                if chunk_idx % 100 == 0:
                    mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
                    print(f"🔄 Chunk {chunk_idx+1}/{len(chunk_manifest)}: {chunk_filename} (Memory: {mem_mb:.1f}MB)")

                # Load ONE chunk only
                with open(chunk_path, 'rb') as f_chunk:
                    chunk_data = pickle.load(f_chunk)
                
                chunk_len = len(chunk_data) if isinstance(chunk_data, (list, tuple)) else 1
                
                # Split this chunk's cases immediately
                chunk_train = []
                chunk_test = []
                
                for case_idx, case in enumerate(chunk_data):
                    global_index = current_case_index + case_idx
                    if global_index in train_indices_set:
                        chunk_train.append(case)
                    else:
                        chunk_test.append(case)
                
                # IMMEDIATELY write to disk and free memory
                if chunk_train:
                    self._append_to_pickle(train_file, chunk_train)
                    train_count += len(chunk_train)
                
                if chunk_test:
                    self._append_to_pickle(test_file, chunk_test)
                    test_count += len(chunk_test)
                
                # Free memory immediately
                del chunk_data, chunk_train, chunk_test
                import gc
                gc.collect()
                
                current_case_index += chunk_len
                
                # Progress update
                if chunk_idx % 500 == 0 and chunk_idx > 0:
                    print(f"✅ Processed {chunk_idx} chunks: {train_count} train, {test_count} test cases")
            
            print(f"✅ Final split: {train_count} train, {test_count} test cases")
            
            # Save labels (these are small)
            for label_type, labels in label_dict.items():
                U.save_info(os.path.join(save_dir, f'train_ys_{label_type}.pkl'), labels[train_index])
                U.save_info(os.path.join(save_dir, f'test_ys_{label_type}.pkl'), labels[test_index])
                
            topology = self.get_topology()
            U.save_info(os.path.join(save_dir, 'topology.pkl'), topology)
            
            if self.config['heterogeneous']:
                edge_types = self.get_edge_types()
                U.save_info(os.path.join(save_dir, 'edge_types.pkl'), edge_types)

        except Exception as e:
            print(f"❌ Streaming processing failed: {e}")
            raise

    def _append_to_pickle(self, filepath, new_data):
        """Append data to pickle file (create if doesn't exist)"""
        if os.path.exists(filepath):
            # Load existing, append, save back
            with open(filepath, 'rb') as f:
                existing = pickle.load(f)
            existing.extend(new_data)
            with open(filepath, 'wb') as f:
                pickle.dump(existing, f)
        else:
            # Create new file
            with open(filepath, 'wb') as f:
                pickle.dump(new_data, f)
            # ...existing code...
    def get_label(self, label_type, run_table):
        """ process() 中调用，用来获取label
        参数
        ----------
        label_type: str
            label的类型，可选：['service', 'anomaly_type']
        run_table: pd.DataFrame

        返回值
        ----------
        labels: torch.tensor()
            label列表
        """
        # Return numpy array of integer labels for the given label_type.
        # Robust handling:
        #   - If the entire column is NaN: assign '[normal]'
        #   - If partially NaN: fill those with '[unknown]'
        #   - Placeholders are ordered to appear last in the label index.
        if label_type not in run_table.columns:
            raise ValueError(f"Missing column '{label_type}' in run_table. Columns: {list(run_table.columns)}")

        col = run_table[label_type]

        if col.isna().all():
            print(f"[warn] Column '{label_type}' all NaN; assigning '[normal]'.")
            col = pd.Series(["[normal]"] * len(col), index=col.index)
        else:
            if col.isna().any():
                print(f"[info] Column '{label_type}' had {int(col.isna().sum())} NaNs -> filled with '[unknown]'.")
            col = col.fillna("[unknown]")

        col = col.astype(str)

        unique = sorted(set(col))
        placeholders = ["[unknown]", "[normal]"]
        ordered = [u for u in unique if u not in placeholders]
        for ph in placeholders:
            if ph in unique:
                ordered.append(ph)

        labels_idx = {v: i for i, v in enumerate(ordered)}
        labels = col.map(labels_idx).to_numpy()
        # Save service label list for reverse mapping later
        if label_type == 'service':
            try:
                os.makedirs(self.config['save_dir'], exist_ok=True)
                U.save_info(os.path.join(self.config['save_dir'], 'service_label_list.pkl'), ordered)
            except Exception as e:
                print(f"[warn] could not save service_label_list.pkl: {e}")
        return labels

    def get_topology(self):
        """ process() 中调用，用来获取topology
        """
        dataset = self.config['dataset']
        if self.config['heterogeneous']:
            # 异质图
            if dataset == 'gaia':
                topology = (
                [8, 6, 8, 4, 6, 4, 2, 9, 1, 3, 3, 7, 1, 7, 5, 0, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0,
                 1, 1, 4, 4, 5, 5, 2, 2, 3, 3, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7,
                 6, 7, 6, 7],
                [6, 8, 4, 8, 4, 6, 9, 2, 3, 1, 7, 3, 7, 1, 0, 5, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7,
                 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5,
                 2, 2, 3, 3])
            elif dataset == '20aiops':
                topology = (
                [2, 3, 4, 5, 6, 7, 8, 9, 13, 10, 11, 12, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6,
                 7, 8, 9, 13, 13, 10, 10, 11, 11, 12, 12, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 0,
                 0, 0, 0, 4, 5, 2, 6, 3, 7, 5, 9],
                [2, 3, 4, 5, 6, 7, 8, 9, 13, 10, 11, 12, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 1, 6, 7, 8, 9, 0,
                 0, 0, 0, 4, 5, 2, 6, 3, 7, 5, 9, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 7, 8,
                 9, 13, 13, 10, 10, 11, 11, 12, 12])
            elif dataset == '21aiops':
                topology = (
                    [12, 12, 13, 13, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 9, 9, 10, 10, 11, 11, 8, 8, 9, 9, 10, 10, 11, 11, 
                     2, 2, 2, 2, 3, 3, 3, 3, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 0, 1, 8, 
                     9, 10, 11, 2, 3, 14, 15, 16, 17, 0, 1, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 6, 4, 6, 4, 6, 4, 6, 4, 
                     2, 3, 2, 3, 2, 3, 2, 3, 14, 15, 16, 17, 14, 15, 16, 17, 7, 7, 7, 7, 5, 5, 5, 5, 2, 2, 2, 2, 3, 3, 
                     3, 3, 0, 1, 8, 9, 10, 11, 2, 3, 14, 15, 16, 17],
                    [0, 1, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 6, 4, 6, 4, 6, 4, 6, 4, 2, 3, 2, 3, 2, 3, 2, 3, 14, 15, 16,
                     17, 14, 15, 16, 17, 7, 7, 7, 7, 5, 5, 5, 5, 2, 2, 2, 2, 3, 3, 3, 3, 0, 1, 8, 9, 10, 11, 2, 3, 14, 15,
                     16, 17, 12, 12, 13, 13, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 9, 9, 10, 10, 11, 11, 8, 8, 9, 9, 10, 10, 11, 11,
                     2, 2, 2, 2, 3, 3, 3, 3, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 0, 1, 8, 9, 10,
                     11, 2, 3, 14, 15, 16, 17]
                )
            else:
                raise Exception()
        else:
            # 同质图
            if dataset == 'gaia':
                topology = (
                    [8, 6, 8, 4, 9, 2, 0, 5, 3, 1, 3, 7, 1, 7, 6, 4, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0,
                     0,
                     1, 1, 4, 4, 5, 5, 2, 2, 3, 3],
                    [6, 8, 4, 8, 2, 9, 5, 0, 1, 3, 7, 3, 7, 1, 4, 6, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6,
                     7,
                     6, 7, 6, 7, 6, 7, 6, 7, 6, 7])  # 正向
            #                 topology = ([8, 6, 8, 4, 6, 4, 2, 9, 1, 3, 3, 7, 1, 7, 5, 0, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7],
            #                            [6, 8, 4, 8, 4, 6, 9, 2, 3, 1, 7, 3, 7, 1, 0, 5, 6, 7, 6, 7, 4, 5, 4, 5, 2, 3, 2, 3, 0, 1, 0, 1, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8, 9, 9, 8, 8, 9, 9, 8, 8, 9, 9, 2, 2, 3, 3, 0, 0, 1, 1, 4, 4, 5, 5, 2, 2, 3, 3])  # 使用异质图
            elif dataset == '20aiops':
                # topology = (
                #     [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 13, 13, 13, 10, 10, 11, 11, 12, 12, 10, 11, 12],
                #     [1, 2, 6, 7, 8, 9, 1, 3, 6, 7, 8, 9, 1, 4, 6, 7, 8, 9, 1, 5, 6, 7, 8, 9, 0, 6, 0, 7, 0, 8, 0, 9, 4, 5, 13, 2, 6, 3, 7, 5, 9, 10, 11, 12])  # 正向
                topology = (
                    [1, 2, 6, 7, 8, 9, 1, 3, 6, 7, 8, 9, 1, 4, 6, 7, 8, 9, 1, 5, 6, 7, 8, 9, 0, 6, 0, 7, 0, 8, 0, 9, 4,
                     5, 13, 2, 6, 3, 7, 5, 9, 10, 11, 12],
                    [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 13,
                     13, 13, 10, 10, 11, 11, 12, 12, 10, 11, 12])  # 反向
            elif dataset == '21aiops':
                topology = ([12, 12, 13, 13, 0, 0, 0, 0, 1, 1, 1, 1, 8, 8, 9, 9, 10, 10, 11, 11, 8, 8, 9, 9, 10, 10, 11, 
                             11, 2, 2, 2, 2, 3, 3, 3, 3, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 14, 15, 16, 17, 
                             0, 1, 8, 9, 10, 11, 2, 3, 14, 15, 16, 17, 12, 13],
                            [0, 1, 0, 1, 8, 9, 10, 11, 8, 9, 10, 11, 6, 4, 6, 4, 6, 4, 6, 4, 2, 3, 2, 3, 2, 3, 2, 3, 14, 
                             15, 16, 17, 14, 15, 16, 17, 7, 7, 7, 7, 5, 5, 5, 5, 2, 2, 2, 2, 3, 3, 3, 3, 0, 1, 8, 9, 10, 
                             11, 2, 3, 14, 15, 16, 17, 12, 13])  # 正向
            else:
                raise Exception()
        return topology

    def get_edge_types(self):
        dataset = self.config['dataset']
        if not self.config['heterogeneous']:
            raise Exception()
        if dataset == 'gaia':
            etype = tensor(np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2]).astype(np.int64))
        elif dataset == '20aiops':
            etype = tensor(np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                 2, 2, 2, 2, 2, 2, 2, 2]).astype(np.int64))
        elif dataset == '21aiops':
            etype = tensor(np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).astype(np.int64))
        else:
            raise Exception()
        return etype

class UnircaLab():
    def __init__(self, config):
        self.config = config
        instances = config['nodes'].split()
        self.ins_dict = dict(zip(instances, range(len(instances))))
        self.demos = pd.read_csv(os.path.join(self.config['data_dir'], self.config['run_table']), index_col=0)
        # Load service label list if present for reverse mapping in testing
        self.service_label_list = None
        svc_map_file = os.path.join(self.config['save_dir'], 'service_label_list.pkl')
        if os.path.exists(svc_map_file):
            try:
                self.service_label_list = U.load_info(svc_map_file)
            except Exception as e:
                print(f"[warn] failed to load service_label_list.pkl: {e}")
        if config['dataset'] == 'gaia':
            self.topoinfo = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9]}
        elif config['dataset'] == '21aiops':
            self.topoinfo = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9, 10, 11], 5: [12, 13], 6: []}
        elif config['dataset'] == '20aiops':
            self.topoinfo = {0: [0, 1], 1: list(range(2, 10)), 2: list(range(10, 14))}
        else:
            raise Exception('Unknow dataset')

    def _get_device(self):
        """Smart device selection with DGL compatibility check"""
        if torch.cuda.is_available():
            try:
                # Test if DGL supports CUDA
                test_graph = dgl.graph(([0], [1]))
                test_graph.to('cuda:0')
                return 'cuda:0'
            except Exception as e:
                print(f"[info] CUDA torch found but DGL GPU build missing: {e}. Using CPU.")
                return 'cpu'
        return 'cpu'
    
    def to_cpu_np(self, tensor):
        """Convert tensor to CPU numpy array safely"""
        return tensor.detach().cpu().numpy()

    def collate(self, samples):
        """OPTIMIZED: Efficient batch collation with memory management."""
        graphs, labels = map(list, zip(*samples))
        
        # Efficient DGL batching
        batched_graph = dgl.batch(graphs)
        
        # Optimized tensor creation with explicit dtype
        batched_labels = torch.tensor(labels, dtype=torch.long)
        
        # Memory cleanup for large batches
        if len(graphs) > 100:  # For large batches, cleanup immediately
            del graphs, labels
            import gc
            gc.collect()
        
        return batched_graph, batched_labels

    def save_result(self, save_path, data):
        df = pd.DataFrame(data, columns=['top_k', 'accuracy'])
        df.to_csv(save_path, index=False)
    
    def train(self, dataset, key):
        # def hook(module, input, output):
        #     features.append(output)
        #     return None
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        # print('len train_dataset=', len(dataset))
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=self.collate)
        device = self._get_device()
        print(f"[device] train using {device} (seed: {self.config.get('seed', 'None')})")
        # print(device)

        in_dim = dataset.graphs[0].ndata['attr'].shape[1]
#         out_dim = len(set([i.item() for i in dataset.labels]))
        out_dim = self.config[key]
        # hid_dim = (in_dim + out_dim) * 2 // 3
        hid_dim = int(np.sqrt(in_dim*out_dim))
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model = RGCNClassifier(in_dim, hid_dim, out_dim, etype).to(device)  # @ 异质图
#             model = RGCNv2Classifier(in_dim, hid_dim, out_dim, etype).to(device)
            # 钩子函数钩取中间结果
#             for (name, module) in model.named_modules():
#                 print("name: ", name)
#             model.conv2.dropout.register_forward_hook(hook)
        else:
#             model = GCNClassifier(in_dim, hid_dim, out_dim).to(device)  # 同质图
#             model = GATClassifier(in_dim, hid_dim, out_dim, 3).to(device) # GAT
#             model = SAGEClassifier(in_dim, hid_dim, out_dim).to(device) # GraphSAGE
#             model = TAGClassifier(in_dim, hid_dim, out_dim) # TAGConv
#             model = GATv2Classifier(in_dim, hid_dim, out_dim, 3).to(device)
#             model = LinearClassifier(in_dim, hid_dim, out_dim).to(device)
#             model = ChebClassifier(in_dim, hid_dim, out_dim, 2, True).to(device) # ChebConv
            model = TAGClassifier(in_dim, hid_dim, out_dim).to(device)
        print(model)

        opt = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model.train()
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata['attr'].float()
                logits = model(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            if len(losses) > self.config['win_size'] and \
                    abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
                # 保存钩子函数的中间结果
#                 with open('feature_out.pkl', 'wb') as f:
#                     pickle.dump(features, f)
                break

        # loss曲线
#         plt.plot(range(len(losses)), losses)
#         plt.show()
        return model
    
    def multi_trainv2(self, dataset_ts, dataset_ta, dataset_t3):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        
        weight = 0.5
        device = self._get_device()
        print(f"[device] multi_trainv2 using {device} (seed: {self.config.get('seed', 'None')})")

        dataloader_ts = DataLoader(dataset_ts, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_t3 = DataLoader(dataset_t3, batch_size=self.config['batch_size'], collate_fn=self.collate)

        in_dim_ts = dataset_ts.graphs[0].ndata['attr'].shape[1]
        out_dim_ts = self.config['N_S']
        hid_dim_ts = (in_dim_ts + out_dim_ts) * 2 // 3
        in_dim_ta = dataset_ta.graphs[0].ndata['attr'].shape[1]
        out_dim_ta = self.config['N_A']
        hid_dim_ta = (in_dim_ta + out_dim_ta) * 2 // 3
        in_dim_t3 = dataset_t3.graphs[0].ndata['attr'].shape[1]
        out_dim_t3 = 2
        hid_dim_t3 = (in_dim_t3 + out_dim_t3) * 2 // 3

        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_ts = RGCNMSL(in_dim_ts, hid_dim_ts, out_dim_ts, etype).to(device)  # @ 异质图
            model_ta = RGCNClassifier(in_dim_ta, hid_dim_ta, out_dim_ta, etype).to(device)
        else:
            model_ts = SGCCClassifier(in_dim_ts, hid_dim_ts, out_dim_ts).to(device)
            model_ta = SGCCClassifier(in_dim_ta, hid_dim_ta, out_dim_ta).to(device)
        print(model_ts)
        print(model_ta)
        
        opt_ts = torch.optim.Adam(model_ts.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        opt_ta = torch.optim.Adam(model_ta.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_ts.train()
        model_ta.train()
        
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for i in range(len(ts_samples)):
                # service
                ts_bg = ts_samples[i][0].to(device)
                ts_labels = ts_samples[i][1].to(device)
                ts_feats = ts_bg.ndata['attr'].float()
                ts_logits = model_ts(ts_bg, ts_feats)
                ts_loss = F.cross_entropy(ts_logits, ts_labels)
                # anomaly_type
                ta_bg = ta_samples[i][0].to(device)
                ta_labels = ta_samples[i][1].to(device)
                ta_feats = ta_bg.ndata['attr'].float()
                ta_logits = model_ta(ta_bg, ta_feats)
                ta_loss = F.cross_entropy(ta_logits, ta_labels)
                
                opt_ts.zero_grad()
                opt_ta.zero_grad()
                
                total_loss = weight*ts_loss+(1-weight)*ta_loss
                total_loss.backward()
                opt_ts.step()
                opt_ta.step()
                epoch_loss += total_loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            # if len(losses) > self.config['win_size'] and \
            #         abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
            #     break
        return model_ts, model_ta  

    def multi_train(self, dataset_ts, dataset_ta):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        weight = 0.5
        device = self._get_device()
        print(f"[device] multi_train using {device} (seed: {self.config.get('seed', 'None')})")
        dataloader_ts = DataLoader(dataset_ts, batch_size=self.config['batch_size'], collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=self.config['batch_size'], collate_fn=self.collate)
        in_dim = dataset_ts.graphs[0].ndata['attr'].shape[1]
        hid_dim = in_dim * 2 // 3
        out_dim_ts = self.config['N_S']
        out_dim_ta = self.config['N_A']
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model = RGCNMSL(in_dim, hid_dim, out_dim_ts, out_dim_ta, etype).to(device)  # @ 异质图
        else:
            raise Exception("haven't set")
        print(model)
        
        opt = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model.train()
        
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for i in range(len(ts_samples)):
                # 两个任务输入的拓扑、特征一致
                bg = ts_samples[i][0].to(device) 
                feats = bg.ndata['attr'].float()
                ts_labels = ts_samples[i][1].to(device)
                ta_labels = ta_samples[i][1].to(device)
                
                ts_logits, ta_logits  = model(bg, feats)
                ta_loss = F.cross_entropy(ta_logits, ta_labels)
                ts_loss = F.cross_entropy(ts_logits, ts_labels)
                
                opt.zero_grad()
                
                total_loss = weight*ts_loss+(1-weight)*ta_loss
                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            # if len(losses) > self.config['win_size'] and \
            #         abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
            #     break
        return model

    def multi_trainv0(self, dataset_ts, dataset_ta):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        weight = 0.5
        device = self._get_device()
        print(f"[device] OPTIMIZED multi_trainv0 using {device} (seed: {self.config.get('seed', 'None')})")
        
        # Get optimization settings from config
        use_mixed_precision = self.config.get('use_mixed_precision', True)
        gradient_clip = self.config.get('gradient_clip', 1.0)
        num_workers = self.config.get('num_workers', 8)
        pin_memory = self.config.get('pin_memory', True)
        prefetch_factor = self.config.get('prefetch_factor', 8)
        persistent_workers = self.config.get('persistent_workers', True) and num_workers > 0
        
        print(f"🚀 Optimizations: Mixed Precision={use_mixed_precision}, Gradient Clip={gradient_clip}")
        print(f"   DataLoader: workers={num_workers}, pin_memory={pin_memory}, prefetch={prefetch_factor}")
        
        # OPTIMIZED DataLoaders with performance enhancements
        dataloader_ts = DataLoader(
            dataset_ts, 
            batch_size=self.config['batch_size'], 
            collate_fn=self.collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            persistent_workers=persistent_workers,
            drop_last=True  # Consistent batch sizes for cuDNN optimization
        )
        dataloader_ta = DataLoader(
            dataset_ta, 
            batch_size=self.config['batch_size'], 
            collate_fn=self.collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            persistent_workers=persistent_workers,
            drop_last=True
        )
        
        in_dim_ts = dataset_ts.graphs[0].ndata['attr'].shape[1]
        out_dim_ts = self.config['N_S']
        # OPTIMIZED hidden dimension ratio from config
        hid_dim_ratio = self.config.get('hidden_dim_ratio', 0.6)
        hid_dim_ts = int((in_dim_ts + out_dim_ts) * hid_dim_ratio)
        
        in_dim_ta = dataset_ta.graphs[0].ndata['attr'].shape[1]
        out_dim_ta = self.config['N_A']
        hid_dim_ta = int((in_dim_ta + out_dim_ta) * hid_dim_ratio)
        
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_ts = RGCNClassifier(in_dim_ts, hid_dim_ts, out_dim_ts, etype).to(device)
            model_ta = RGCNClassifier(in_dim_ta, hid_dim_ta, out_dim_ta, etype).to(device)
        else:
            model_ts = TAGClassifier(in_dim_ts, hid_dim_ts, out_dim_ts).to(device)
            model_ta = TAGClassifier(in_dim_ta, hid_dim_ta, out_dim_ta).to(device)
        print(model_ts)
        print(model_ta)
        
        # OPTIMIZED AdamW optimizers with advanced settings
        opt_ts = torch.optim.AdamW(
            model_ts.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            amsgrad=True
        )
        opt_ta = torch.optim.AdamW(
            model_ta.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['weight_decay'],
            betas=(0.9, 0.999),
            amsgrad=True
        )
        
        # Cosine Annealing LR Schedulers
        scheduler_ts = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ts, 
            T_max=self.config['epoch'],
            eta_min=self.config['lr'] * 0.01
        )
        scheduler_ta = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_ta, 
            T_max=self.config['epoch'],
            eta_min=self.config['lr'] * 0.01
        )
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and 'cuda' in device else None
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config.get('win_size', 50)
        
        losses = []
        model_ts.train()
        model_ta.train()
        
        print(f"🚀 Starting OPTIMIZED streaming training: {len(dataset_ts)} samples, {self.config['epoch']} epochs")
        print(f"   Using {'Mixed Precision' if scaler else 'Full Precision'} training")
        
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            
            # Create fresh iterators for each epoch to enable streaming
            dataloader_ts_iter = iter(dataloader_ts)
            dataloader_ta_iter = iter(dataloader_ta)
            
            # Process batches one at a time without pre-loading
            try:
                while True:
                    # Get next batch from streaming dataloaders
                    ts_batch = next(dataloader_ts_iter)
                    ta_batch = next(dataloader_ta_iter)
                    
                    # Move to device with non_blocking for performance
                    ts_bg = ts_batch[0].to(device, non_blocking=True)
                    ts_labels = ts_batch[1].to(device, non_blocking=True)
                    ta_bg = ta_batch[0].to(device, non_blocking=True)
                    ta_labels = ta_batch[1].to(device, non_blocking=True)
                    
                    opt_ts.zero_grad()
                    opt_ta.zero_grad()
                    
                    # MIXED PRECISION forward pass
                    if scaler:
                        with torch.cuda.amp.autocast():
                            # Service prediction
                            ts_feats = ts_bg.ndata['attr'].float()
                            ts_logits = model_ts(ts_bg, ts_feats)
                            ts_loss = F.cross_entropy(ts_logits, ts_labels)
                            
                            # Anomaly type prediction
                            ta_feats = ta_bg.ndata['attr'].float()
                            ta_logits = model_ta(ta_bg, ta_feats)
                            ta_loss = F.cross_entropy(ta_logits, ta_labels)
                            
                            total_loss = weight * ts_loss + (1 - weight) * ta_loss
                        
                        # Scaled backward pass
                        scaler.scale(total_loss).backward()
                        
                        # Gradient clipping with scaler
                        if gradient_clip > 0:
                            scaler.unscale_(opt_ts)
                            scaler.unscale_(opt_ta)
                            torch.nn.utils.clip_grad_norm_(model_ts.parameters(), gradient_clip)
                            torch.nn.utils.clip_grad_norm_(model_ta.parameters(), gradient_clip)
                        
                        scaler.step(opt_ts)
                        scaler.step(opt_ta)
                        scaler.update()
                    else:
                        # Standard precision
                        ts_feats = ts_bg.ndata['attr'].float()
                        ts_logits = model_ts(ts_bg, ts_feats)
                        ts_loss = F.cross_entropy(ts_logits, ts_labels)
                        
                        ta_feats = ta_bg.ndata['attr'].float()
                        ta_logits = model_ta(ta_bg, ta_feats)
                        ta_loss = F.cross_entropy(ta_logits, ta_labels)
                        
                        total_loss = weight * ts_loss + (1 - weight) * ta_loss
                        total_loss.backward()
                        
                        # Gradient clipping
                        if gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model_ts.parameters(), gradient_clip)
                            torch.nn.utils.clip_grad_norm_(model_ta.parameters(), gradient_clip)
                        
                        opt_ts.step()
                        opt_ta.step()
                    
                    epoch_loss += total_loss.detach().item()
                    epoch_cnt += 1
                    
            except StopIteration:
                # End of epoch - both dataloaders are exhausted
                pass
            
            # Step schedulers
            scheduler_ts.step()
            scheduler_ta.step()
            
            # Calculate average epoch loss
            avg_epoch_loss = epoch_loss / epoch_cnt if epoch_cnt > 0 else 0
            losses.append(avg_epoch_loss)
            
            # Early stopping check
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"🛑 Early stopping at epoch {epoch+1}/{self.config['epoch']}")
                    break
            
            # Progress reporting every 10 epochs
            if (epoch + 1) % 10 == 0:
                current_lr = scheduler_ts.get_last_lr()[0]
                print(f"Epoch {epoch+1}/{self.config['epoch']}: Loss={avg_epoch_loss:.4f}, LR={current_lr:.2e}")
        
        print("🎉 OPTIMIZED training completed!")
        return model_ts, model_ta  
        
    def trans_train(self, dataset_src, dataset_target, retrain=False):
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
        dataloader_src = DataLoader(dataset_src, batch_size=self.config['batch_size'], collate_fn=self.collate)
        device = self._get_device()
        print(f"[device] trans_train using {device} (seed: {self.config.get('seed', 'None')})")

        in_dim = dataset_src.graphs[0].ndata['attr'].shape[1]
        out_dim = self.config['N_A']
#         hid_dim = (in_dim + out_dim) * 2 // 3
        hid_dim = in_dim * 2 // 3
        if self.config['heterogeneous']:
            etype = U.load_info(os.path.join(self.config['save_dir'], 'edge_types.pkl'))
            model_src = RGCNClassifier(in_dim, hid_dim, out_dim, etype).to(device)  # @ 异质图
        else:
            model_src = SGCCClassifier(in_dim, hid_dim, out_dim).to(device)
        print(model_src)

        opt = torch.optim.Adam(model_src.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_src.train()
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader_src:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata['attr'].float()
                logits = model_src(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            # if len(losses) > self.config['win_size'] and \
            #         abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
            #     break
        # 至此源模型训练完成，开始目标模型迁移
        model_target = copy.deepcopy(model_src)
        print('retrain: ', retrain)
        if not retrain: # 是否重新训练模型
            for p in model_target.parameters():
                p.requires_grad = False
        dataloader_target = DataLoader(dataset_target, batch_size=self.config['batch_size'], collate_fn=self.collate)
        in_dim = dataset_target.graphs[0].ndata['attr'].shape[1]
        out_dim = self.config['N_S']
        hid_dim = in_dim * 2 // 3
        # 将最后一层替换为新的全连接层，其余层保留，新添加的层默认requires_grad=True
        model_target.classify = nn.Linear(hid_dim, out_dim)
        print(model_target)
        # 重新训练
        opt = torch.optim.Adam(model_target.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        losses = []
        model_target.train()
        for epoch in tqdm(range(self.config['epoch'])):
            epoch_loss = 0
            epoch_cnt = 0
            features = []
            for batched_graph, labels in dataloader_target:
                batched_graph = batched_graph.to(device)
                labels = labels.to(device)
                feats = batched_graph.ndata['attr'].float()
                logits = model_target(batched_graph, feats)
                loss = F.cross_entropy(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.detach().item()
                epoch_cnt += 1
            losses.append(epoch_loss / epoch_cnt)
            # if len(losses) > self.config['win_size'] and \
            #         abs(losses[-self.config['win_size']] - losses[-1]) < self.config['win_threshold']:
            #     break
        
        return model_target
    
    # 获取训练集和测试集的编码
    def get_embedings(self, model, train_dataset, test_dataset):
        model.eval()
        trainloader = DataLoader(train_dataset, batch_size=len(train_dataset) + 10, collate_fn=self.collate)
        testloader = DataLoader(test_dataset, batch_size=len(test_dataset) + 10, collate_fn=self.collate)
        for batched_graph, labels in trainloader:
            train_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float())
        
        for batched_graph, labels in testloader:
            test_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float())
        dataset = self.config['dataset']
        with open(f'results/{dataset}_train_embeds.pkl', 'wb') as f:
            pickle.dump(train_embeds, f)
        with open(f'results/{dataset}_test_embeds.pkl', 'wb') as f:
            pickle.dump(test_embeds, f)
        return
    
    def test_cls(self, model, train_dataset, test_dataset, classifier, task):
        model.eval()
        trainloader = DataLoader(train_dataset, batch_size=len(train_dataset) + 10, collate_fn=self.collate)
        testloader = DataLoader(test_dataset, batch_size=len(test_dataset) + 10, collate_fn=self.collate)
        for batched_graph, labels in trainloader:
            train_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float(), True)
            classifier.fit(train_embeds.detach().cpu().numpy(), labels.detach().cpu().numpy())
        
        for batched_graph, labels in testloader:
            test_embeds = model.get_embeds(batched_graph, batched_graph.ndata['attr'].float(), True)
            # Move to CPU for sklearn
            output = classifier.predict_proba(test_embeds.detach().cpu().numpy())
            labels = labels.detach().cpu().numpy().reshape(-1, 1)
            # print(classifier.score(test_embeds.detach().numpy(), labels))
            preds = [
                [
                    item[-1] for item in sorted(list(zip(output[i], range(len(output[i]))))[: 5], reverse=True)
                    ] for i in range(len(output))
                ]
            if task == 'instance':
                ser_res = pd.DataFrame(np.append(preds, labels, axis=1), columns=
                                       np.append([f'Top{i}' for i in range(1, len(preds[0])+1)], 'GroundTruth'))
                self.test_instance_local(ser_res, max_num=2)
            elif task == 'anomaly_type':
                preds = np.array(preds)
                pre = precision_score(labels, preds[:, 0], average='weighted')
                rec = recall_score(labels, preds[:, 0], average='weighted')
                f1 = f1_score(labels, preds[:, 0], average='weighted')
                print('Weighted precision', pre)
                print('Weighted recall', rec)
                print('Weighted f1-score', f1)
            else:
                raise Exception('Unknow task')
            
        return
    
    def testv2(self, model, dataset, task, out_file, save_file=None):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset) + 10, collate_fn=self.collate)
        device = self._get_device()
        print(f"[device] testv2 using {device} (seed: {self.config.get('seed', 'None')})")
        seed = self.config['seed']
        accuracy = []
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            output = model(batched_graph, batched_graph.ndata['attr'].float())
            k = 5 if output.shape[-1] >= 5 else output.shape[-1]
            if task == 'instance':
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)  
                out_dir = os.path.join(self.config['save_dir'], 'preds')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy().reshape(-1, 1)
                ser_res = pd.DataFrame(np.append(y_pred, y_true, axis=1), 
                                       columns=np.append([f'Top{i}' for i in range(1, len(y_pred[0])+1)], 'GroundTruth'))
                
                # 定位到实例级别
                accs, ins_res = self.test_instance_local(ser_res, max_num=2)
                ins_res.to_csv(f'{out_dir}/multitask_seed{seed}_{out_file}')
                columns = ['A@1', 'A@2', 'A@3', 'A@4', 'A@5']
            elif task == 'anomaly_type':
                _, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)  
                out_dir = os.path.join(self.config['save_dir'], 'preds')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                y_pred = indices.detach().cpu().numpy()
                y_true = labels.detach().cpu().numpy().reshape(-1, 1)
                pre = precision_score(y_pred[:, 0], y_true, average='weighted')
                rec = recall_score(y_pred[:, 0], y_true, average='weighted')
                f1 = f1_score(y_pred[:, 0], y_true, average='weighted')
                print('Weighted precision', pre)
                print('Weighted recall', rec)
                print('Weighted f1-score', f1)
                # test_cases = self.demos[self.demos['data_type']=='test'] # This line is not needed for saving

                # --- FIX: Remove index=test_cases.index ---
                output_df = pd.DataFrame(np.append(
                    y_pred[:, 0].reshape(-1, 1), y_true, axis=1), 
                    columns=['Pred', 'GroundTruth']) 
                
                output_csv_path = f'{out_dir}/multitask_seed{seed}_{out_file}'
                # Save without specifying the incorrect index, and don't write pandas index to CSV
                output_df.to_csv(output_csv_path, index=False) 
                print(f"✅ Saved anomaly type predictions to: {output_csv_path}")
                # --- END FIX ---

                columns = ['Precision', 'Recall', 'F1-Score']
                accs = np.array([pre, rec, f1])
            else:
                raise Exception('Unknow task')

        if save_file:
            accuracy = pd.DataFrame(accs.reshape(-1, len(columns)), columns=columns)
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file.split('_')[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

        return output, labels
    
    def test(self, model, dataset, out_file, save_file=None):
        model.eval()
        dataloader = DataLoader(dataset, batch_size=len(dataset) + 10, collate_fn=self.collate)
        device = self._get_device()
        print(f"[device] test using {device} (seed: {self.config.get('seed', 'None')})")
        seed = self.config['seed']
        accuracy = []
        for batched_graph, labels in dataloader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)
            if self.config['heterogeneous']:
                output = model(batched_graph, batched_graph.ndata['attr'].float())
            else:
                output = model(batched_graph, batched_graph.ndata['attr'].float())
            for k in range(1, 6):
                values, indices = torch.topk(output, k=k, dim=1, largest=True, sorted=True)
                # 保存Top5的预测结果
                if k == 5:
                    out_dir = os.path.join(self.config['save_dir'], 'preds')
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    y_pred = indices.detach().cpu().numpy()
                    y_true = labels.detach().cpu().numpy().reshape(-1, 1)
                    pd.DataFrame(np.append(y_pred, y_true, axis=1), columns=['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth']).to_csv(f'{out_dir}/seed{seed}_{out_file}')
                num = 0
                for i in range(len(indices)):
                    num += indices[i].eq(labels[i]).sum().item()
                print(f'top{k} acc: ', num / len(indices))
                accuracy.append([k, num / len(indices)])

        if save_file:
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file.split('_')[0])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

        return output, labels

    def test_multitask(self, model, dataset_ts, dataset_ta, out_file, save_file_ts=None, save_file_ta=None):
        model.eval()
        dataloader_ts = DataLoader(dataset_ts, batch_size=len(dataset_ts) + 10, collate_fn=self.collate)
        dataloader_ta = DataLoader(dataset_ta, batch_size=len(dataset_ta) + 10, collate_fn=self.collate)
        ts_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ts]
        ta_samples = [(batched_graphs, labels) for batched_graphs, labels in dataloader_ta]
        device = self._get_device()
        print(f"[device] test_multitask using {device} (seed: {self.config.get('seed', 'None')})")
        seed = self.config['seed']
        accuracy_ts = []
        accuracy_ta = []
        for i in range(len(ts_samples)):
            batched_graph = ts_samples[i][0].to(device)
            labels_ts = ts_samples[i][1].to(device)
            labels_ta = ta_samples[i][1].to(device)
            output_ts, output_ta = model(batched_graph, batched_graph.ndata['attr'].float())
            print('service')
            for k in range(1, 6):
                _, indices_ts = torch.topk(output_ts, k=k, dim=1, largest=True, sorted=True)
                
                # 保存Top5的根因微服务组定位预测结果---->实例定位结果
                if k == 5:
                    out_dir = os.path.join(self.config['save_dir'], 'preds')
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    y_pred = indices_ts.detach().cpu().numpy()
                    y_true = labels_ts.detach().cpu().numpy().reshape(-1, 1)
                    # pd.DataFrame(np.append(y_pred, y_true, axis=1), columns=['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth']).to_csv(f'{out_dir}/multitask_seed{seed}_{out_file}')
                    ser_res = pd.DataFrame(np.append(y_pred, y_true, axis=1), columns=
                                           np.append([f'Top{i}' for i in range(1, len(y_pred[0])+1)], 'GroundTruth'))
                    # 定位到实例级别
                    print('instance')
                    _, ins_res = self.test_instance_local(ser_res, 2)
                    ins_res.to_csv(f'{out_dir}/multitask_seed{seed}_{out_file}')
                      
                # num_ts = 0
                # for i in range(len(indices_ts)):
                #     num_ts += indices_ts[i].eq(labels_ts[i]).sum().item()
                # print(f'top{k} acc: ', num_ts / len(indices_ts))
                # accuracy_ts.append([k, num_ts / len(indices_ts)])
                
            print('anomaly type') # anomaly type需要求pre、rec、f1
            for k in range(1, 6):
                _, indices_ta = torch.topk(output_ta, k=k, dim=1, largest=True, sorted=True)
                num_ta = 0
                for i in range(len(indices_ta)):
                    num_ta += indices_ta[i].eq(labels_ta[i]).sum().item()
                print(f'top{k} acc: ', num_ta / len(indices_ta))
                accuracy_ts.append([k, num_ta / len(indices_ta)])

        # if save_file_ts:
        #     save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file_ts.split('_')[0])
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     self.save_result(f'{save_dir}/seed{seed}_{save_file_ts}', accuracy_ts)
        # if save_file_ta:
        #     save_dir = os.path.join(self.config['save_dir'], 'evaluations', save_file_ta.split('_')[0])
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     self.save_result(f'{save_dir}/seed{seed}_{save_file_ta}', accuracy_ta)

        return

    # ... locate def test_instance_local(self, s_preds, max_num=2): and replace its body ...

    def test_instance_local(self, s_preds, max_num=2):
        """
        根据微服务的预测结果预测微服务的根因实例
        Robust: skips rows with NaN instance; handles short ins_pred lists.
        """
        txt_path = self.config['text_path']
        if not os.path.isabs(txt_path):
            txt_path = os.path.join(self.config['data_dir'], txt_path)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"stratification_texts file not found: {txt_path}")
        with open(txt_path, 'rb') as f:
            info = pickle.load(f)

        test_cases = self.demos[self.demos['data_type'] == 'test']

        # Collect results only for valid rows
        topks = np.zeros(5, dtype=float)
        ins_preds = []
        gt_indices = []
        valid_case_indices = []

        # Pre-calc key type and fast instance word count cache
        i = 0
        for idx, row in test_cases.iterrows():
            # Skip if instance missing
            inst_name = row.get('instance', None)
            if pd.isna(inst_name):
                # skip this test row
                i += 1
                continue
            # Skip if instance label not in mapping
            if inst_name not in self.ins_dict:
                # optional: dynamically add -> uncomment next two lines if desired
                # self.ins_dict[inst_name] = len(self.ins_dict)
                # (but then topology / downstream metrics may misalign)
                i += 1
                continue

            # info key casting
            # Some datasets use datetime / numeric keys
            info_key_type = type(list(info.keys())[0])
            try:
                cast_idx = info_key_type(idx)
            except Exception:
                cast_idx = idx
            if cast_idx not in info:
                i += 1
                continue

            # Build word-count dict for this case
            num_dict = {}
            for pair in info[cast_idx]:
                # pair: (instance, anomalyFlag) or similar
                inst_key = pair[0]
                if inst_key in self.ins_dict:
                    num_dict[self.ins_dict[inst_key]] = len(str(info[cast_idx][pair]).split())

            s_pred_row = s_preds.loc[i]  # same positional order the caller built
            ins_pred = []

            # Iterate predicted service columns (exclude GroundTruth)
            for col in list(s_preds.columns)[:-1]:
                svc_idx = s_pred_row[col]

                # Skip NaN service predictions
                if pd.isna(svc_idx):
                    continue

                # If predicted service index is integer and exists directly
                if isinstance(svc_idx, (int, np.integer)) and svc_idx not in self.topoinfo:
                    # Try mapping via service_label_list if stored
                    if self.service_label_list and 0 <= svc_idx < len(self.service_label_list):
                        # Here we just keep numeric; no direct name mapping in topology
                        pass
                    else:
                        continue  # cannot map
                if svc_idx not in self.topoinfo:
                    continue

                candidates = self.topoinfo[svc_idx]
                if not candidates:
                    continue
                # Score candidates by num_dict counts (missing -> 0)
                temp = sorted(
                    [(ins_id, num_dict.get(ins_id, 0)) for ins_id in candidates],
                    key=lambda x: x[1],
                    reverse=True
                )
                ins_pred.extend([item[0] for item in temp[:max_num]])

            # Keep only up to 5 predictions (pad or truncate)
            ins_pred = ins_pred[:5]
            if len(ins_pred) < 5:
                # pad with -1 so indexing below is safe
                ins_pred = ins_pred + [-1] * (5 - len(ins_pred))

            # Record
            ins_preds.append(ins_pred)
            gt_idx = self.ins_dict[inst_name]
            gt_indices.append(gt_idx)
            valid_case_indices.append(idx)

            # Update top-k accuracy curve
            for k in range(5):
                if ins_pred[k] == gt_idx:
                    topks[k:] += 1
                    break

            i += 1  # move to next original s_preds row

        total = len(gt_indices)
        if total == 0:
            print("[warn] No valid test rows (all instances missing or unmapped).")
            return np.zeros(5), pd.DataFrame(columns=['Top1','Top2','Top3','Top4','Top5','GroundTruth'])

        topk_acc = topks / total
        print('Top1-5: ', topk_acc)

        y_true = np.array(gt_indices).reshape(-1, 1)
        df = pd.DataFrame(
            np.append(np.array(ins_preds), y_true, axis=1),
            columns=['Top1', 'Top2', 'Top3', 'Top4', 'Top5', 'GroundTruth'],
            index=valid_case_indices
        )
        return topk_acc, df    
    
    def cross_evaluate(self, s_output, s_labels, a_output, a_labels, save_file=None):
        N_S = self.config['N_S']
        N_A = self.config['N_A']
        TOPK_SA = self.config['TOPK_SA']
        # softmax取正（使用笛卡尔积比大小）
        s_values = nn.Softmax(dim=1)(s_output)
        a_values = nn.Softmax(dim=1)(a_output)
        # 获得 K_ * K_的笛卡尔积
        product = []
        for k in range(len(s_values)):
            service_val = s_values[k]
            anomaly_val = a_values[k]
            m = torch.zeros(N_S * N_A).reshape(N_S, N_A)
            for i in range(N_S):
                for j in range(N_A):
                    m[i][j] = service_val[i] * anomaly_val[j]
            product.append(m)
        # 获得每个笛卡尔积矩阵的topk及坐标
        sa_topks = []
        for idx in range(len(product)):
            m = product[idx]
            topk = []
            last_max_val = 1
            for k in range(TOPK_SA):
                cur_max_val = tensor(0)
                x = 0
                y = 0
                for i in range(N_S):
                    for j in range(N_A):
                        if m[i][j] > cur_max_val and m[i][j] < last_max_val:
                            cur_max_val = m[i][j]
                            x = i
                            y = j
                topk.append(((x, y), cur_max_val.item()))
                last_max_val = cur_max_val
            sa_topks.append(topk)

        # 使用笛卡尔积计算分数得到service + anomaly_type 的topk结果
        accuracy = []
        for k in range(1, TOPK_SA + 1):
            num = 0
            for i in range(len(s_labels)):
                label = (s_labels[i].item(), a_labels[i].item())
                predicts = sa_topks[i][:k]
                for predict in predicts:
                    if predict[0] == label:
                        num += 1
                        break
            print(f'top{k} acc: ', num / len(s_labels))
            accuracy.append([k, num / len(s_labels)])
        if save_file:
            seed = self.config['seed']
            save_dir = os.path.join(self.config['save_dir'], 'evaluations', 'service_anomaly')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_result(f'{save_dir}/seed{seed}_{save_file}', accuracy)

    # Inside class UnircaLab in He_DGL.py
    # Inside class UnircaLab:
    def do_lab(self, lab_id):
        save_dir = os.path.join(self.config['save_dir'], str(lab_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Update config's save_dir to point to the specific lab_id folder
        self.config['save_dir'] = save_dir 
        RawDataProcess(self.config).process()
        
        # Check label diversity to warn about meaningless training
        try:
            import dgl.data.utils as U
            train_service_labels = U.load_info(os.path.join(save_dir, 'train_ys_service.pkl'))
            train_anomaly_labels = U.load_info(os.path.join(save_dir, 'train_ys_anomaly_type.pkl'))
            
            service_unique = len(set(train_service_labels))
            anomaly_unique = len(set(train_anomaly_labels))
            
            if service_unique == 1 and anomaly_unique == 1:
                print(f"⚠️  WARNING: Both service ({service_unique} unique) and anomaly_type ({anomaly_unique} unique) have only 1 class.")
                print("Training will converge instantly with meaningless results.")
                print("Consider loading proper labels or skipping training.")
                
            print(f"📊 Label diversity: {service_unique} services, {anomaly_unique} anomaly types")
        except Exception as e:
            print(f"[info] Could not check label diversity: {e}")
        # 训练
        s = time.time()
        print('train starts at', s)
        
        # 分别训练模型 (Keep original comments)
#       service_model = self.train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                 os.path.join(save_dir, 'train_ys_service.pkl'),
#                                                 os.path.join(save_dir, 'topology.pkl'),
#                                                 aug=self.config['aug'],
#                                                 aug_size=self.config['aug_size'],
#                                                 shuffle=True), 'N_S')
                                                
#       anomaly_type_model = self.train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                       os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
#                                                       os.path.join(save_dir, 'topology.pkl'),
#                                                       aug=self.config['aug'],
#                                                       aug_size=self.config['aug_size'],
#                                                       shuffle=True), 'N_A')

#       trans_model = self.trans_train(UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                       os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
#                                                       os.path.join(save_dir, 'topology.pkl'),
#                                                       aug=self.config['aug'],
#                                                       aug_size=self.config['aug_size'],
#                                                       shuffle=True), 
#                                       UnircaDataset(os.path.join(save_dir, 'train_Xs.pkl'),
#                                                       os.path.join(save_dir, 'train_ys_service.pkl'),
#                                                       os.path.join(save_dir, 'topology.pkl'),
#                                                       aug=self.config['aug'],
#                                                       aug_size=self.config['aug_size'],
#                                                       shuffle=True),
#                                       retrain=True)

        # --- CORRECTED: Determine training dataset path (streaming or traditional) ---
        placeholder_path_raw = os.path.join(save_dir, 'train_Xs_streaming.pkl') 
        streaming_placeholder = os.path.normpath(placeholder_path_raw) # Normalize the path

        print(f"DEBUG do_lab: Checking for streaming placeholder at: {streaming_placeholder}") # Optional Debug Print

        if os.path.exists(streaming_placeholder):
            # Load streaming mode configuration
            try:
                with open(streaming_placeholder, 'rb') as f:
                     streaming_config = pickle.load(f)
                # Use the chunked directory path STORED IN the placeholder
                xs_path = os.path.normpath(streaming_config['chunked_dir']) # Normalize this path too
                print(f"📦 do_lab: Using STREAMING mode. Dataset path (chunked dir): {xs_path}")
            except Exception as e:
                print(f"[ERROR] Failed to read streaming config from placeholder: {e}")
                xs_path = os.path.normpath(self.config['Xs']) # Fallback (Normalized)
                # Construct expected chunked dir path from base config path for fallback
                xs_path = f"{xs_path}_chunked" # Or _chunks depending on previous fix
                xs_path = os.path.normpath(xs_path) 
                print(f"[WARNING] Using fallback constructed chunked dir path: {xs_path}")
        else:
            # Use traditional .pkl files created in the lab_id specific save_dir
            print(f"📁 do_lab: Placeholder NOT found at '{streaming_placeholder}'. Using TRADITIONAL mode.")
            xs_path_raw = os.path.join(save_dir, 'train_Xs.pkl')
            xs_path = os.path.normpath(xs_path_raw) # Normalize
            print(f"   Dataset path: {xs_path}")
        # --- End Path Determination ---

        # --- Call multi_trainv0 ---
        # --- REMOVED incorrect 'config=self.config' arguments ---
        model_ts, model_ta = self.multi_trainv0(UnircaDataset(xs_path, # <-- Pass correct path
                                                              os.path.join(save_dir, 'train_ys_service.pkl'),
                                                              os.path.join(save_dir, 'topology.pkl'),
                                                              # config=self.config, <-- REMOVED
                                                              aug=self.config['aug'],
                                                              aug_size=self.config['aug_size'],
                                                              shuffle=True), 
                                              UnircaDataset(xs_path, # <-- Pass correct path
                                                              os.path.join(save_dir, 'train_ys_anomaly_type.pkl'),
                                                              os.path.join(save_dir, 'topology.pkl'),
                                                              # config=self.config, <-- REMOVED
                                                              aug=self.config['aug'],
                                                              aug_size=self.config['aug_size'],
                                                              shuffle=True))
        # --- END REMOVAL ---

        t1 = time.time()
        print('train ends at', t1)
        print('train use time', t1 - s, 's') 
        
        # 测试并分析准确率
        s = time.time()
        print('test starts at', s)
#       print('[Training respectively]')
#       print('instance')
        
        # --- CORRECTED: Determine test dataset path (streaming or traditional) ---
        test_placeholder_path_raw = os.path.join(save_dir, 'test_Xs_streaming.pkl')
        test_streaming_placeholder = os.path.normpath(test_placeholder_path_raw) # Normalize

        if os.path.exists(test_streaming_placeholder):
             # Use same chunked dir as training (already determined as xs_path)
             test_xs_path = xs_path 
             print(f"📦 do_lab: Using STREAMING mode for test data: {test_xs_path}")
        else:
             test_xs_path_raw = os.path.join(save_dir, 'test_Xs.pkl')
             test_xs_path = os.path.normpath(test_xs_path_raw) # Normalize
             print(f"📁 do_lab: Using TRADITIONAL mode for test data: {test_xs_path}")
        # --- End Path Determination ---

        print('instance')
        # --- REMOVED incorrect 'config=self.config' arguments ---
        _, _ = self.testv2(model_ts,
                          UnircaDataset(test_xs_path, # <-- USE CORRECT TEST PATH
                                        os.path.join(save_dir, 'test_ys_service.pkl'),
                                        os.path.join(save_dir, 'topology.pkl')),
                                        # config=self.config), <-- REMOVED
                          'instance',
                          'instance_pred_multi_v0.csv',
                          'instance_acc_multi_v0.csv')
        print('anomaly type')
        # --- REMOVED incorrect 'config=self.config' arguments ---
        _, _ = self.testv2(model_ta,
                          UnircaDataset(test_xs_path, # <-- USE CORRECT TEST PATH
                                        os.path.join(save_dir, 'test_ys_anomaly_type.pkl'),
                                        os.path.join(save_dir, 'topology.pkl')),
                                        # config=self.config), <-- REMOVED
                          'anomaly_type',
                          'anomaly_pred_multi_v0.csv',
                          'anomaly_acc_multi_v0.csv')
        # --- END REMOVAL ---
        
        # print('multi_task learning')
        # self.test_multitask(model_m,  UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
        #                                              os.path.join(save_dir, 'test_ys_service.pkl'),
        #                                              os.path.join(save_dir, 'topology.pkl')), 
        #                                UnircaDataset(os.path.join(save_dir, 'test_Xs.pkl'),
        #                                              os.path.join(save_dir, 'test_ys_anomaly_type.pkl'),
        #                                              os.path.join(save_dir, 'topology.pkl')), 
        #                                'service_pred_multi.csv', 
        #                                'service_acc_multi.csv', 
        #                                'anomaly_type_acc_multi.csv')

        t = time.time()
        print('test ends at', t)
        print('test use time', t - s, 's')
        # 保存模型
        if self.config['save_model']:
            torch.save(model_ts, os.path.join(save_dir, 'service_model.pt'))
            torch.save(model_ta, os.path.join(save_dir, 'anomaly_type_model.pt'))