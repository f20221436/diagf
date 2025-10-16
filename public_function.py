import os
import pickle
import argparse
import yaml


def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def save_chunked(file, data, chunk_size=1000):
    """Save large data in chunks to avoid memory issues"""
    from tqdm import tqdm  # Add missing import
    import tempfile
    import shutil
    
    print(f"💾 Saving {len(data)} items in chunks of {chunk_size}...")
    
    # Prepare metadata
    base_path = file.replace('.pkl', '')
    metadata = {
        'total_cases': len(data),
        'services_per_case': len(data[0]) if data else 0,
        'embedding_dim': len(data[0][0]) if data and data[0] else 0,
        'chunks': []
    }
    
    # Save data in chunks
    for i in tqdm(range(0, len(data), chunk_size), desc="Saving chunks"):
        chunk = data[i:i + chunk_size]
        chunk_path = f"{base_path}_chunk_{i//chunk_size:04d}.pkl"
        
        # Save chunk
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Add to metadata
        metadata['chunks'].append({
            'path': chunk_path,
            'start_idx': i,
            'end_idx': min(i + chunk_size, len(data)),
            'size': len(chunk)
        })
        
        # Clear chunk from memory
        del chunk
    
    # Save metadata file
    metadata_path = f"{base_path}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Create a dummy main file (for compatibility)
    with open(file, 'wb') as f:
        pickle.dump({'chunked': True, 'metadata_path': metadata_path}, f)
    
    print(f"✅ Saved {len(metadata['chunks'])} chunks")
    print(f"📁 Metadata: {metadata_path}")
    print(f"📁 Main file: {file}")

def load_chunked(file):
    """
    Load data that was saved with save_chunked.
    """
    from tqdm import tqdm
    
    with open(file, 'rb') as f:
        # Load metadata first
        metadata = pickle.load(f)
        
        if metadata.get('data_type') != 'chunked_embeddings':
            # Fallback to regular load for non-chunked files
            f.seek(0)
            return pickle.load(f)
        
        # Load chunks
        data = []
        total_chunks = (metadata['total_items'] + metadata['chunk_size'] - 1) // metadata['chunk_size']
        
        for _ in tqdm(range(total_chunks), desc="Loading chunks", unit="chunk"):
            chunk = pickle.load(f)
            data.extend(chunk)
            
        print(f"✅ Loaded {len(data)} items from chunked file")
        return data


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()
    with open(os.path.join('./config', args.config), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def min_max_normalized(feature):
    feature_copy = feature.copy().astype(float)
    for i in range(len(feature_copy)):
        min_f, max_f = min(feature_copy[i]), max(feature_copy[i])
        if min_f == max_f:
            feature_copy[i] = [0]*len(feature_copy[i])
        else:
            feature_copy[i] = (feature_copy[i] - min_f) / (max_f - min_f)
    return feature_copy


def deal_config(config, key):
    new_config = {}
    for k in config[key].keys():
        if 'path' in k or 'dir' in k:
            if config[key][k] or config[key][k] == '':
                path = os.path.join(config['base_path'], config['demo_path'],
                                    config['label'], config[key][k])
                if 'dir' in k:
                    if not os.path.exists(path):
                        os.makedirs(path)
                new_config[k] = path
            else:
                new_config[k] = config[key][k]
        else:
            new_config[k] = config[key][k]

    return new_config

def load_chunked_iter(path):
    """
    Yields data from a file that was saved iteratively or in chunks, 
    one item at a time, without loading the whole file into memory.
    """
    from tqdm import tqdm
    
    with open(path, 'rb') as f:
        # First, try to load metadata to see if it's our special chunked format
        try:
            metadata = pickle.load(f)
            if metadata.get('data_type') != 'chunked_embeddings':
                # If not, assume it's a simple pickle file and we can't iterate
                f.seek(0) # Rewind the file
                print("Warning: Not a chunked file, attempting to load all at once.")
                yield pickle.load(f)
                return
        except (pickle.UnpicklingError, EOFError):
            # This is likely a file saved one-item-at-a-time, not in chunks
            # Rewind and process item by item
            f.seek(0)

        # Now, load the actual data, one item or one chunk at a time
        try:
            while True:
                chunk = pickle.load(f)
                # If the loaded object is a list (a chunk), yield its items one by one
                if isinstance(chunk, list):
                    for item in chunk:
                        yield item
                else: # If it's a single item
                    yield chunk
        except EOFError:
            # This is the expected way to finish
            pass



if __name__ == '__main__':
    config = get_config()
    print(config['fasttext']['vector_dim'])
    cur_path = os.getcwd()
    print(cur_path[:cur_path.find('unirca')])

