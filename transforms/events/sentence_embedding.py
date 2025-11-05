# diagf/transforms/events/sentence_embedding.py
# CLEANED VERSION - Contains only the correct functions

import os
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import pickle

# Note: public_function as pf is imported by the script that runs this

def read_text(path):
    """Reads lines from a text file, extracting the text before the first tab."""
    print(f"DEBUG read_text: Reading from {path}")
    text = []
    lines_read = 0
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                lines_read += 1
                parts = line.split('\t', 1)
                if parts:
                    # Append even if parts[0] is empty, TFIDF handles empty strings
                    text.append(parts[0]) 
    except FileNotFoundError:
        print(f"[ERROR] read_text: File not found at {path}")
    print(f"DEBUG read_text: Read {lines_read} lines, resulted in {len(text)} text entries.")
    return text

def tfidf_word_embedding(tfidf_matrix, data_dict, texts, word_dict, service_num):
    """Calculates TF-IDF weighted sentence embeddings using sparse matrix."""
    
    if not data_dict or not isinstance(data_dict, dict):
         print("[ERROR] tfidf_word_embedding: Invalid or empty event embedding dictionary (data_dict).")
         return []
         
    first_key = next(iter(data_dict.keys()), None)
    if first_key is None:
        print("[ERROR] tfidf_word_embedding: Event embedding dictionary (data_dict) has no keys.")
        return []
    try:
         # Ensure the value is indexable (list, tuple, ndarray)
         if hasattr(data_dict[first_key], '__len__'):
              length = len(data_dict[first_key]) # Dimension of embeddings
         else:
              raise ValueError("First value in data_dict is not a sequence.")
    except Exception as e:
         print(f"[ERROR] tfidf_word_embedding: Could not determine embedding dimension from data_dict['{first_key}']: {e}")
         return []

    if length == 0:
        print("[ERROR] tfidf_word_embedding: Embedding dimension is zero.")
        return []

    sentence_embedding = []
    case_embedding = []
    
    total_texts = len(texts)
    print(f"DEBUG tfidf_word_embedding: Processing {total_texts} text entries...")
    
    # Check if number of texts matches expected structure
    if total_texts > 0 and service_num > 0 and total_texts % service_num != 0:
        remainder = len(texts) % service_num
        if remainder != 0:
            print(f"[INFO] Dropping {remainder} extra lines to align with service_num={service_num}.")
            texts = texts[:len(texts) - remainder]
            print(f"[INFO] After trimming: {len(texts)} text entries (perfectly divisible by {service_num}).")
    
    for count, text in enumerate(tqdm(texts, desc="Generating Embeddings")):
        temp = np.zeros(length, dtype=np.float32) # Initialize embedding vector for this sentence
        
        if text and isinstance(text, str): # Ensure text is a non-empty string
            words = text.split(' ') # Split into words/events
            unique_words = list(set(w for w in words if w)) # Get unique non-empty words
            
            try:
                # Get the sparse TF-IDF row efficiently
                if count < tfidf_matrix.shape[0]:
                    sparse_row = tfidf_matrix.getrow(count)
                else:
                    # print(f"[Warning] Index {count} out of bounds for tfidf_matrix shape {tfidf_matrix.shape}. Skipping.")
                    continue

                for word in unique_words:
                    if word in word_dict and word in data_dict: # Check both dictionaries
                        word_index_in_tfidf = word_dict[word]
                        
                        # Check bounds before accessing sparse matrix column
                        if word_index_in_tfidf < sparse_row.shape[1]:
                           weight = sparse_row[0, word_index_in_tfidf]
                           word_vector = data_dict[word]
                           
                           # Ensure vector is a numpy array and has the correct length
                           if not isinstance(word_vector, np.ndarray):
                               word_vector = np.array(word_vector, dtype=np.float32)
                           
                           if word_vector.shape == (length,): # Check shape consistency
                              temp += weight * word_vector
                           # else: # Optional warning for shape mismatch
                           #    print(f"[Warning] Shape mismatch for word '{word}'. Expected ({length},), got {word_vector.shape}. Skipping.")
                        # else: # Optional: Warn if word index is out of bounds
                        #    print(f"[Warning] Word '{word}' index {word_index_in_tfidf} out of bounds for TF-IDF row.")

            except IndexError:
                 # print(f"[Warning] IndexError accessing TF-IDF matrix for text index {count}. Skipping.")
                 continue
            except Exception as e:
                 print(f"[Error] Unexpected error processing text index {count}: {e}. Skipping.")
                 continue

        # Append the calculated (or zero) embedding for this text/sentence
        case_embedding.append(temp)
        
        # Group embeddings by case (using service_num)
        # Append when we have collected 'service_num' embeddings OR it's the very last text entry
        if service_num > 0 and ((count + 1) % service_num == 0 or (count + 1) == total_texts):
            if case_embedding: # Only append if the list is not empty
                # If it's the last entry and case_embedding is incomplete, pad it?
                # For now, append as is. Downstream might need handling.
                if len(case_embedding) != service_num and (count + 1) == total_texts:
                     print(f"[WARNING] Final case embedding has {len(case_embedding)} services, expected {service_num}.")
                sentence_embedding.append(case_embedding)
            case_embedding = [] # Reset for the next case
            
    # Redundant check, handled in loop now
    # if case_embedding:
    #     print(f"[Warning] Trailing embeddings found ({len(case_embedding)}). Check if service_num aligns.")

    print(f"DEBUG tfidf_word_embedding: Generated {len(sentence_embedding)} case embeddings.")
    return sentence_embedding


def sentence_embedding_main(file_dict_path, train_path, test_path, save_path, service_num, pf_module):
    """Main function to load data, compute embeddings, and save using chunking."""
    print(f"Loading event embeddings from: {file_dict_path}")
    try:
        data_dict = pf_module.load(file_dict_path)
        if not isinstance(data_dict, dict) or not data_dict:
             print("[ERROR] Loaded event embedding file is not a valid dictionary or is empty.")
             return
    except Exception as e:
        print(f"[ERROR] Failed to load event embedding file '{file_dict_path}': {e}")
        return

    print(f"Reading training text from: {train_path}")
    train_text = read_text(train_path)
    print(f"Reading testing text from: {test_path}")
    test_text = read_text(test_path)

    if not train_text and not test_text:
        print("[ERROR] Both train and test text files are empty or could not be read. Cannot generate embeddings.")
        return

    # Initialize TF-IDF
    vectorizer = CountVectorizer(lowercase=False, token_pattern=r"(?u)\b[\w\-_<>]+\b")
    transformer = TfidfTransformer()
    
    # --- Fit TF-IDF only on training data ---
    print("Fitting CountVectorizer and TfidfTransformer on training data...")
    if train_text:
        vec_train = vectorizer.fit_transform(train_text)
        tfidf_train = transformer.fit_transform(vec_train) # Sparse matrix
    else:
        print("[WARNING] Training text is empty. TF-IDF model might be poorly initialized.")
        # Fit with dummy data or handle differently? For now, create empty sparse matrix.
        vec_train = vectorizer.fit_transform([""]) # Fit with a dummy
        tfidf_train = transformer.fit_transform(vec_train)

    # --- Transform test data using the fitted vectorizer ---
    print("Transforming test data...")
    if test_text:
        vec_test = vectorizer.transform(test_text) # Use transform, not fit_transform
        tfidf_test = transformer.transform(vec_test) # Use transform, not fit_transform
    else:
        print("[INFO] Test text is empty.")
        tfidf_test = None # Indicate no test data to process

    word = vectorizer.get_feature_names_out()
    word_dict = {word[i]: i for i in range(len(word))}
    
    print(f'Vectorizer vocabulary size (from train data): {len(word_dict)}')
    print(f'FastText vocabulary size (from event_embedding.pkl): {len(data_dict)}')
    
    # Generate embeddings, passing sparse matrices
    print("\nGenerating training embeddings...")
    train_embedding = []
    if train_text:
        train_embedding = tfidf_word_embedding(tfidf_train, data_dict, train_text, word_dict, service_num)

    print("\nGenerating testing embeddings...")
    test_embedding = []
    if test_text and tfidf_test is not None:
        test_embedding = tfidf_word_embedding(tfidf_test, data_dict, test_text, word_dict, service_num)

    # Combine embeddings
    all_embeddings = train_embedding + test_embedding
    
    if not all_embeddings:
        print("[ERROR] No embeddings were generated. Output file will not be saved.")
        return

    # --- Determine Shape for Metadata ---
    shape_dim1 = 0
    shape_dim2 = 0
    shape_dim3 = 0
    try:
        shape_dim1 = len(all_embeddings)
        if shape_dim1 > 0:
             # Find first non-empty case
             first_non_empty_case = next((case for case in all_embeddings if case), None)
             if first_non_empty_case:
                  shape_dim2 = len(first_non_empty_case) # Services per case
                  if shape_dim2 > 0:
                       # Find the first non-empty embedding vector to get dim3
                       first_vec = next((vec for vec in first_non_empty_case if isinstance(vec, np.ndarray) and vec.size > 0), None)
                       if first_vec is not None:
                            shape_dim3 = len(first_vec) # Embedding dimension
                       else:
                            print("[Warning] Could not find any valid embedding vectors in the first case.")
             else:
                  print("[Warning] All generated case embeddings appear empty.")
        print(f'\nDetermined embedding shape: {shape_dim1} cases × {shape_dim2} services × {shape_dim3} dimensions')
    except Exception as e:
        print(f"\n[Warning] Could not determine full shape of combined embeddings: {e}")

    # --- Use Chunked Saving with Metadata ---
    print(f"\nSaving combined embeddings with chunking to base path: {save_path}")
    chunk_size = 500 # Define chunk size
    try:
        # Use manual chunked saving with metadata
        print(f"[INFO] Using manual chunked pickle saving.")
        # Get base path and ensure .pkl extension for metadata
        base_path, _ = os.path.splitext(save_path) 
        ext = ".pkl" # Force .pkl extension
        
        # Create the dedicated chunk directory
        chunk_dir = f"{base_path}_chunks" # Name folder based on base path
        os.makedirs(chunk_dir, exist_ok=True)
        print(f"   Saving chunks into directory: {chunk_dir}")

        chunk_paths_for_metadata = [] # Store paths relative to metadata or absolute

        for i in range(0, len(all_embeddings), chunk_size):
            chunk_data = all_embeddings[i:i + chunk_size]
            # Save chunks INSIDE the chunk directory
            chunk_filename = f"{os.path.basename(base_path)}_chunk_{i//chunk_size:04d}{ext}"
            chunk_path = os.path.join(chunk_dir, chunk_filename)

            with open(chunk_path, 'wb') as f_chunk:
                pickle.dump(chunk_data, f_chunk, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Store absolute path in metadata for simplicity
            chunk_paths_for_metadata.append({
                'path': os.path.abspath(chunk_path), # Store absolute path
                'start_idx': i,
                'end_idx': min(i + chunk_size, len(all_embeddings)),
                'size': len(chunk_data)
            })

        print(f"   Saved {len(chunk_paths_for_metadata)} chunks.")
        
        # Save metadata file next to the base path (not inside chunks dir)
        metadata_path = f"{base_path}_metadata{ext}" 
        metadata = {
            'total_cases': shape_dim1,
            'services_per_case': shape_dim2,
            'embedding_dim': shape_dim3,
            'chunk_size': chunk_size, 
            'chunks': chunk_paths_for_metadata # Use the detailed list
        }
        with open(metadata_path, 'wb') as f_meta:
             pickle.dump(metadata, f_meta, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   Chunk metadata saved to: {metadata_path}")

    except Exception as e:
        print(f"[ERROR] Failed to save embeddings using chunking: {e}")
        # Optionally add fallback

    # Optional: Clean up memory
    del all_embeddings, train_embedding, test_embedding, data_dict, train_text, test_text
    print("Memory cleanup done.")