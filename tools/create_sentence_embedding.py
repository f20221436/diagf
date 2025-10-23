# diagf/tools/create_sentence_embedding.py
import os
import sys
import argparse

# --- Add necessary paths ---
# Get the directory containing this script (tools)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (diagf)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory (diagf) to the Python path
sys.path.insert(0, parent_dir)

# --- Import project modules (Now Python can find them in 'diagf') ---
try:
    import public_function as pf
    # Import the main function from your module (using correct path from diagf)
    from transforms.events.sentence_embedding import sentence_embedding_main
except ImportError as e:
    print(f"[ERROR] Failed to import required modules: {e}")
    print(f"        Ensure 'public_function.py' is in '{parent_dir}'")
    print(f"        Ensure 'sentence_embedding.py' is in '{os.path.join(parent_dir, 'transforms', 'events')}'")
    sys.exit(1)

# --- Main execution block ---
if __name__ == "__main__":

    # --- 1. DEFINE ABSOLUTE PATHS to your data ---
    # Root folder for the preprocessed GAIA data
    data_root = r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS'
    anomalies_dir = os.path.join(data_root, 'anomalies')
    # Temp dir where fastText created train.txt/test.txt
    fasttext_temp_dir = os.path.join(anomalies_dir, 'fasttext_temp')
    # Location where run_fastText saved event_embedding.pkl (relative to parent_dir 'diagf')
    fasttext_output_dir = os.path.join(parent_dir, 'fasttext')

    # --- 2. Configuration ---
    # Paths to the required input files
    event_embedding_path = os.path.join(fasttext_output_dir, 'event_embedding.pkl')
    # Use train.txt and test.txt from the fasttext_temp directory
    train_txt_path = os.path.join(fasttext_temp_dir, 'train.txt')
    test_txt_path = os.path.join(fasttext_temp_dir, 'test.txt')

    # Path for the final output file (base name for chunking)
    output_embedding_base_path = os.path.join(anomalies_dir, 'sentence_embedding') # NOTE: No .pkl here

    # Number of services/nodes (should match your fastText config)
    num_services = 10 # Based on 'dbservice1 dbservice2 ... webservice2'

    # --- 3. Validation ---
    print("Checking for required input files...")
    missing_files = []
    required_paths = [event_embedding_path, train_txt_path, test_txt_path]
    for p in required_paths:
        if not os.path.exists(p):
            missing_files.append(p)

    # Also check if the base directory for output exists
    output_dir = os.path.dirname(output_embedding_base_path)
    if not os.path.isdir(output_dir):
        print(f"[WARNING] Output directory does not exist, will be created: {output_dir}")

    if missing_files:
        print("[ERROR] Cannot run: Required input files are missing!")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("✅ All input files found.")

    # --- Run the Main Function from the imported module ---
    print("\n🚀 Starting Sentence Embedding Generation...")
    try:
        sentence_embedding_main(
            file_dict_path=event_embedding_path,
            train_path=train_txt_path,
            test_path=test_txt_path,
            save_path=output_embedding_base_path, # Pass the base path
            service_num=num_services,
            pf_module=pf # Pass the imported public_function module
        )
        print(f"\n🎉 Process completed successfully!")
        print(f"   Chunk files saved in: {output_embedding_base_path}_chunks")
        print(f"   Metadata saved to: {output_embedding_base_path}_metadata.pkl")
    except Exception as e:
        print(f"\n❌ An error occurred during embedding generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)