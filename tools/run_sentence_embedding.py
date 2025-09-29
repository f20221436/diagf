import argparse
import pickle
import string
import os
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# =============================================================================
# Helper functions
# =============================================================================

def save(path, data):
    """Saves data to a file using pickle with a more robust, atomic write."""
    temp_path = path + ".tmp"
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(data, f)
        os.rename(temp_path, path)
        print(f"Data successfully saved to {path}")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def load(path):
    """Loads data from a file using pickle."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data successfully loaded from {path}")
    return data

# =============================================================================
# MODIFICATION 1: Worker functions now load data from disk
# =============================================================================

worker_data = {}

def init_worker(source_path, word_dict_path, tfidf_matrix_path):
    """Initializes worker by loading all necessary data from file paths."""
    # Each worker process loads its own copy of the data
    worker_data['data_dict'] = load(source_path)
    worker_data['word_dict'] = load(word_dict_path)
    
    # Create the idx -> word mapping
    word_dict = worker_data['word_dict']
    worker_data['idx2word'] = {i: w for w, i in word_dict.items()}
    
    # Load the sparse matrix
    worker_data['tfidf_csr'] = scipy.sparse.load_npz(tfidf_matrix_path)
    
    # Infer embedding length
    sample_vec = next(iter(worker_data['data_dict'].values()))
    worker_data['emb_len'] = len(sample_vec)


def process_sentence_row(i):
    """Processes a single row identified by its index 'i'."""
    data_dict = worker_data['data_dict']
    idx2word = worker_data['idx2word']
    emb_len = worker_data['emb_len']
    tfidf_csr = worker_data['tfidf_csr']
    
    temp = np.zeros(emb_len, dtype="float32")
    
    # Get pointers to the non-zero elements for row 'i'
    row_start = tfidf_csr.indptr[i]
    row_end = tfidf_csr.indptr[i+1]
    col_indices = tfidf_csr.indices[row_start:row_end]
    tfidf_values = tfidf_csr.data[row_start:row_end]
    
    for j in range(len(col_indices)):
        feat_idx = col_indices[j]
        weight = tfidf_values[j]
        word = idx2word.get(feat_idx)
        
        if word and word in data_dict:
            temp += weight * np.array(data_dict[word], dtype="float32")
            
    return temp

# =============================================================================
# Core logic
# =============================================================================

def read_text(path):
    print(f"   Counting lines in {os.path.basename(path)}...")
    with open(path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    text = []
    print(f"   Reading {total_lines:,} lines from {os.path.basename(path)}...")
    with open(path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc=f"Reading {os.path.basename(path)}", unit="lines", ncols=100):
            line = line.strip()
            if not line:
                continue
            first = line.split('\t')[0]
            text.append(first.strip())
    return text

def tfidf_word_embedding_safe(tfidf_matrix_path, source_path, word_dict_path, num_rows, service_num, n_jobs, desc="Embedding"):
    all_embeddings = []
    print(f"   Processing {num_rows:,} sentences for {desc} using {n_jobs} job(s)...")

    if n_jobs > 1:
        chunksize = max(1, num_rows // (n_jobs * 10))
        init_args = (source_path, word_dict_path, tfidf_matrix_path)
        
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=init_worker, initargs=init_args) as executor:
            row_iterator = range(num_rows)
            all_embeddings = list(tqdm(executor.map(process_sentence_row, row_iterator, chunksize=chunksize), total=num_rows, desc=desc, unit="sent", ncols=100))
    else:
        # Single-threaded mode also loads from disk for consistency
        init_worker(source_path, word_dict_path, tfidf_matrix_path)
        for i in tqdm(range(num_rows), desc=desc, unit="sent", ncols=100):
            temp = process_sentence_row(i)
            all_embeddings.append(temp)
    
    sentence_embedding = []
    if all_embeddings:
        for i in range(0, len(all_embeddings), service_num):
            sentence_embedding.append(all_embeddings[i:i + service_num])

    return sentence_embedding

def sentence_embedding(file_dict, train_path, test_path, save_path, service_num, n_jobs):
    # Create a temporary directory for intermediate files
    temp_dir = "temp_embedding_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Define paths for our temporary files
    temp_word_dict_path = os.path.join(temp_dir, "word_dict.pkl")
    temp_tfidf_train_path = os.path.join(temp_dir, "tfidf_train.npz")
    temp_tfidf_test_path = os.path.join(temp_dir, "tfidf_test.npz")

    try:
        train_text = read_text(train_path)
        test_text = read_text(test_path)

        if not train_text and not test_text:
            raise ValueError("Both train and test text files are empty.")

        vectorizer = CountVectorizer(lowercase=False, token_pattern=r'(?u)\b\S\S+')
        transformer = TfidfTransformer()

        print("   Fitting TF-IDF model...")
        if not train_text:
            vec_corpus = vectorizer.fit_transform(test_text)
            tfidf_test = transformer.fit_transform(vec_corpus)
            tfidf_train = scipy.sparse.csr_matrix((0, tfidf_test.shape[1]))
        else:
            vec_train = vectorizer.fit_transform(train_text)
            tfidf_train = transformer.fit_transform(vec_train)
            print("   Transforming test data...")
            vec_test = vectorizer.transform(test_text)
            tfidf_test = transformer.transform(vec_test)

        word = vectorizer.get_feature_names_out()
        word_dict = {word[i]: i for i in range(len(word))}
        
        print("   Saving temporary data files for workers...")
        save(temp_word_dict_path, word_dict)
        scipy.sparse.save_npz(temp_tfidf_train_path, tfidf_train)
        scipy.sparse.save_npz(temp_tfidf_test_path, tfidf_test)
        
        print(f"📊 Vectorizer vocabulary size: {len(word_dict):,}")

        train_embedding = tfidf_word_embedding_safe(temp_tfidf_train_path, file_dict, temp_word_dict_path, tfidf_train.shape[0], service_num, n_jobs, desc="Train Embeddings")
        test_embedding = tfidf_word_embedding_safe(temp_tfidf_test_path, file_dict, temp_word_dict_path, tfidf_test.shape[0], service_num, n_jobs, desc="Test Embeddings")

        train_embedding.extend(test_embedding)

        if not train_embedding or not train_embedding[0]:
            print("⚠️ Resulting sentence_embedding is empty. Check your input files.")
        else:
            print(f"✅ Final embedding shape approx: {len(train_embedding)} × {len(train_embedding[0])} × {len(train_embedding[0][0])}")

        save(save_path, train_embedding)
        print(f"💾 Saved → {save_path}")

    finally:
        # --- IMPORTANT: Clean up temporary files ---
        print("   Cleaning up temporary files...")
        for path in [temp_word_dict_path, temp_tfidf_train_path, temp_tfidf_test_path]:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(temp_dir):
            try:
                os.rmdir(temp_dir)
            except OSError as e:
                print(f"Could not remove temp directory {temp_dir}. It may contain other files. Error: {e}")

# =============================================================================
# Main execution block
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sentence embeddings using TF-IDF and pre-trained word vectors.")
    # ... (arguments remain the same)
    parser.add_argument('--source', type=str, required=True, help='Path to the event embedding .pkl file.')
    parser.add_argument('--train', type=str, required=True, help='Path to the train logs .txt file.')
    parser.add_argument('--test', type=str, required=True, help='Path to the test logs .txt file.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the final sentence embedding .pkl file.')
    parser.add_argument('--k_s', type=int, required=True, help='Number of services to group together (service_num).')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs to run. Defaults to 1 (no parallelism).')
    
    args = parser.parse_args()
    
    if args.n_jobs <= 1:
        print("Running in single-threaded mode.")
    else:
        print(f"🚀 Starting sentence embedding process with {args.n_jobs} parallel jobs...")

    sentence_embedding(
        file_dict=args.source,
        train_path=args.train,
        test_path=args.test,
        save_path=args.output,
        service_num=args.k_s,
        n_jobs=args.n_jobs
    )
    print("✅ Process finished successfully.")