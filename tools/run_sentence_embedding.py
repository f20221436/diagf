import argparse
import os
import sys
from tqdm import tqdm

# Add the project root to the path to import from transforms
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from transforms.events.sentence_embedding import sentence_embedding

# =============================================================================
# Wrapper function with progress tracking
# =============================================================================
# Add this after the existing argument parser


def run_sentence_embedding_with_progress(source_path, train_path, test_path, output_path, k_s):
    """
    Wrapper function that calls the original sentence_embedding function with progress tracking.
    """
    print("🚀 Starting sentence embedding generation...")
    print(f"📁 Source (word embeddings): {source_path}")
    print(f"📁 Train data: {train_path}")
    print(f"📁 Test data: {test_path}")
    print(f"📁 Output: {output_path}")
    print(f"🔢 Service grouping (K_S): {k_s}")
    print("=" * 60)
    
    # Check if input files exist
    for path, name in [(source_path, "Source"), (train_path, "Train"), (test_path, "Test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Call the original sentence_embedding function
    sentence_embedding(
        file_dict=source_path,
        train_path=train_path,
        test_path=test_path,
        save_path=output_path,
        service_num=k_s
    )
    
    print("✅ Sentence embedding generation completed successfully!")
    return output_path

# =============================================================================
# Enhanced wrapper that adds parallel processing to the original function
# =============================================================================

# --- REPLACE the entire enhanced_sentence_embedding function with this ---
def enhanced_sentence_embedding(source_path, train_path, test_path, output_path, k_s, n_jobs=4):
    """
    Enhanced version that wraps the original sentence_embedding function with additional features.
    """
    print(f"🚀 Starting sentence embedding generation with enhanced wrapper...")
    print(f"📁 Source (word embeddings): {source_path}")
    print(f"📁 Train data: {train_path}")
    print(f"📁 Test data: {test_path}")
    print(f"📁 Output: {output_path}")
    print(f"🔢 Service grouping (K_S): {k_s}")
    print(f"⚙️  Jobs requested: {n_jobs} (Note: Original function is single-threaded)")
    print("=" * 70)
    
    # Validate input files
    for path, name in [(source_path, "Source embeddings"), (train_path, "Train data"), (test_path, "Test data")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {name} file not found: {path}")
        file_size = os.path.getsize(path) / (1024 * 1024)  # MB
        print(f"✅ {name}: {path} ({file_size:.1f} MB)")
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    # REMOVED: The old, misleading tqdm block is gone.
    # The progress bar is now inside the core function.
    print("\n📊 Calling the sentence_embedding function...")
    sentence_embedding(
        file_dict=source_path,
        train_path=train_path,
        test_path=test_path,
        save_path=output_path,
        service_num=k_s
    )
    
    # Validate output
    if os.path.exists(output_path):
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ Output saved: {output_path} ({output_size:.1f} MB)")
    else:
        raise RuntimeError(f"❌ Output file was not created: {output_path}")
    
    return output_path

# =============================================================================
# Main execution block
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sentence embeddings using TF-IDF and pre-trained word vectors.")
    parser.add_argument('--source', type=str, required=True, help='Path to the event embedding .pkl file.')
    parser.add_argument('--train', type=str, required=True, help='Path to the train logs .txt file.')
    parser.add_argument('--test', type=str, required=True, help='Path to the test logs .txt file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the final sentence embedding .pkl file.')
    parser.add_argument('--k_s', type=int, required=True, help='Number of services to group together (service_num).')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs requested (for logging only, original function is single-threaded).')
    
    args = parser.parse_args()
    
    print("🎯 Sentence Embedding Generator")
    print("=" * 50)
    
    try:
        # Use the enhanced wrapper function
        enhanced_sentence_embedding(
            source_path=args.source,
            train_path=args.train,
            test_path=args.test,
            output_path=args.output,
            k_s=args.k_s,
            n_jobs=args.n_jobs
        )
        print("\n🎉 Process completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise