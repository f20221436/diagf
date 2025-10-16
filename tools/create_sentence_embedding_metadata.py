import os
import sys
import pickle
import glob
from tqdm import tqdm
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

import public_function as pf

def extract_chunks_from_sentence_embedding(input_path, base_output_path, chunk_size=500):
    """
    Extract chunks from existing sentence_embedding.pkl and save as separate files
    """
    print(f"📂 Loading sentence embeddings from: {input_path}")
    
    # Load the embeddings using chunked loader
    try:
        embeddings = pf.load_chunked(input_path)
        print(f"✅ Loaded {len(embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return False
    
    # Get dimensions
    if embeddings and len(embeddings) > 0:
        services_per_case = len(embeddings[0]) if embeddings[0] else 0
        embedding_dim = len(embeddings[0][0]) if embeddings[0] and embeddings[0] else 0
        print(f"📊 Structure: {len(embeddings)} cases × {services_per_case} services × {embedding_dim} dimensions")
    else:
        print(f"❌ No embeddings found or invalid structure")
        return False
    
    # Create chunks and save as separate files
    print(f"💾 Creating chunks of size {chunk_size}...")
    
    chunks_info = []
    total_cases = len(embeddings)
    
    for i in tqdm(range(0, total_cases, chunk_size), desc="Creating chunk files"):
        chunk_data = embeddings[i:i + chunk_size]
        chunk_path = f"{base_output_path}_chunk_{i//chunk_size:04d}.pkl"
        
        # Save chunk
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Record chunk info
        chunks_info.append({
            'path': chunk_path,
            'start_idx': i,
            'end_idx': min(i + chunk_size, total_cases),
            'size': len(chunk_data)
        })
    
    # Create metadata
    metadata = {
        'total_cases': total_cases,
        'services_per_case': services_per_case,
        'embedding_dim': embedding_dim,
        'chunks': chunks_info
    }
    
    # Save metadata
    metadata_path = f"{base_output_path}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Successfully created {len(chunks_info)} chunk files!")
    print(f"📁 Chunks: {base_output_path}_chunk_XXXX.pkl")
    print(f"📁 Metadata: {metadata_path}")
    
    return metadata_path

def create_metadata_for_existing_chunks(base_path, chunk_size=500):
    """
    Create metadata file for existing chunked sentence embeddings or extract chunks from single file.
    
    Args:
        base_path (str): Base path without .pkl extension 
                        (e.g., 'data/gaia/demo/demo_1100/anomalies/sentence_embedding')
        chunk_size (int): Size of chunks if extracting from single file
    
    Returns:
        str: Path to created metadata file
    """
    
    print(f"🔍 Looking for chunk files with pattern: {base_path}_chunk_*.pkl")
    
    # Find all chunk files
    chunk_pattern = f"{base_path}_chunk_*.pkl"
    chunk_files = sorted(glob.glob(chunk_pattern))
    
    # If no chunk files found, try to extract from single file
    if not chunk_files:
        print(f"❌ No chunk files found matching pattern: {chunk_pattern}")
        
        # Check if single sentence_embedding.pkl exists
        single_file = f"{base_path}.pkl"
        if os.path.exists(single_file):
            print(f"✅ Found single embedding file: {single_file}")
            print(f"🔄 Extracting chunks from single file...")
            
            # Extract chunks from the single file
            return extract_chunks_from_sentence_embedding(single_file, base_path, chunk_size)
        else:
            print(f"❌ Neither chunk files nor single file found.")
            print(f"   Expected single file: {single_file}")
            print(f"   Please check if the base path is correct.")
            return None
    
    print(f"✅ Found {len(chunk_files)} chunk files")
    
    # Initialize metadata
    chunks = []
    total_cases = 0
    services_per_case = 0
    embedding_dim = 0
    
    print("📊 Analyzing chunk files...")
    
    # Process each chunk to build metadata
    for i, chunk_file in enumerate(tqdm(chunk_files, desc="Analyzing chunks")):
        try:
            # Load chunk to get its properties
            with open(chunk_file, 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Get chunk info
            chunk_size = len(chunk_data)
            start_idx = total_cases
            end_idx = total_cases + chunk_size
            
            # Get dimensions from first chunk with data
            if i == 0 and chunk_data and len(chunk_data) > 0:
                if isinstance(chunk_data[0], list) and len(chunk_data[0]) > 0:
                    services_per_case = len(chunk_data[0])
                    if isinstance(chunk_data[0][0], list) and len(chunk_data[0][0]) > 0:
                        embedding_dim = len(chunk_data[0][0])
                    else:
                        embedding_dim = 0
                        print(f"⚠️  Warning: Unexpected embedding structure in first chunk")
                else:
                    print(f"⚠️  Warning: Unexpected chunk structure in first chunk")
            
            # Add chunk info to metadata
            chunks.append({
                'path': chunk_file,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'size': chunk_size
            })
            
            total_cases += chunk_size
            
            # Clear chunk from memory
            del chunk_data
            
        except Exception as e:
            print(f"❌ Error processing chunk {chunk_file}: {e}")
            return None
    
    # Create metadata dictionary
    metadata = {
        'total_cases': total_cases,
        'services_per_case': services_per_case,
        'embedding_dim': embedding_dim,
        'chunks': chunks
    }
    
    # Save metadata file
    metadata_path = f"{base_path}_metadata.pkl"
    
    try:
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n✅ Successfully created metadata file!")
        print(f"📁 Metadata file: {metadata_path}")
        print(f"📊 Total cases: {total_cases:,}")
        print(f"📊 Services per case: {services_per_case}")
        print(f"📊 Embedding dimension: {embedding_dim}")
        print(f"📊 Number of chunks: {len(chunks)}")
        
        return metadata_path
        
    except Exception as e:
        print(f"❌ Error saving metadata file: {e}")
        return None

def validate_metadata(metadata_path):
    """
    Validate the created metadata file by checking if it can be loaded
    and contains expected structure.
    """
    try:
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        required_keys = ['total_cases', 'services_per_case', 'embedding_dim', 'chunks']
        missing_keys = [key for key in required_keys if key not in metadata]
        
        if missing_keys:
            print(f"⚠️  Metadata validation warning: Missing keys {missing_keys}")
            return False
        
        # Check if chunk files exist
        missing_chunks = []
        for chunk_info in metadata['chunks']:
            if not os.path.exists(chunk_info['path']):
                missing_chunks.append(chunk_info['path'])
        
        if missing_chunks:
            print(f"⚠️  Metadata validation warning: Missing chunk files:")
            for chunk in missing_chunks[:5]:  # Show first 5
                print(f"    - {chunk}")
            if len(missing_chunks) > 5:
                print(f"    ... and {len(missing_chunks) - 5} more")
            return False
        
        print(f"✅ Metadata validation passed!")
        return True
        
    except Exception as e:
        print(f"❌ Metadata validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Create metadata file for existing chunked sentence embeddings or extract chunks from single file"
    )
    parser.add_argument(
        '--base-path', 
        required=True,
        help='Base path without .pkl extension (e.g., data/gaia/demo/demo_1100/anomalies/sentence_embedding)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Chunk size when extracting from single file (default: 500)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate the created metadata file'
    )
    
    args = parser.parse_args()
    
    print("🚀 Sentence Embedding Metadata Creator & Chunk Extractor")
    print("=" * 60)
    
    # Create metadata (will extract chunks if needed)
    metadata_path = create_metadata_for_existing_chunks(args.base_path, args.chunk_size)
    
    if metadata_path:
        # Validate if requested
        if args.validate:
            print("\n🔍 Validating metadata...")
            validate_metadata(metadata_path)
        
        print(f"\n🎉 Process completed successfully!")
        print(f"📁 Metadata file created: {metadata_path}")
        print(f"You can now run your training with the metadata file available.")
    else:
        print(f"\n❌ Failed to create metadata file.")
        exit(1)

if __name__ == "__main__":
    main()