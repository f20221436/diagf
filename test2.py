import pickle
import numpy as np
import os
import sys

# --- IMPORTANT ---
# You MUST update this variable to the full, exact path of your .pkl file.
# The path you gave (C:\Users\...\diagf\fasttext) looks like a directory.
# You need to add the actual filename.
#
# For example, if your file is named 'fasttext_embeddings.pkl' and
# is in that directory, the path should be:
# r"C:\Users\DEVESH PALO\projects\DiagFusionWorking\diagf\fasttext\fasttext_embeddings.pkl"
#
# --- SET YOUR FILE PATH HERE ---

FILE_PATH = r"C:\Users\DEVESH PALO\projects\DiagFusionWorking\diagf\fasttext\event_embedding.pkl"

# ---------------------------------

def inspect_pickle_file(file_path):
    """
    Loads a pickle file and prints a summary of its contents,
    expecting a dictionary of {token: numpy_vector}.
    """
    print(f"--- Inspecting Pickle File ---")

    # 1. Check if file exists and get size
    try:
        if not os.path.exists(file_path):
            print(f"\n[Error] File not found at path:")
            print(f"{file_path}")
            print("\nPlease double-check the FILE_PATH variable in this script.")
            return
            
        file_size_kb = os.path.getsize(file_path) / 1024
        print(f"File:      {os.path.basename(file_path)}")
        print(f"Location:  {os.path.dirname(file_path)}")
        print(f"Size:      {file_size_kb:.2f} KB")

        if file_size_kb < 1:
            print("[Warning] File is very small. It might be empty or incomplete.")
            
    except Exception as e:
        print(f"[Error] Could not get file stats: {e}")
        return

    # 2. Try to load the file
    try:
        with open(file_path, 'rb') as f:
            # Use encoding='latin1' for good compatibility with pickle files
            # containing numpy arrays, especially across different Python/numpy versions.
            data = pickle.load(f, encoding='latin1') 
        print(f"\n--- File Loaded Successfully ---")
        
    except pickle.UnpicklingError:
        print("\n[Error] File is corrupted or not a valid pickle file.")
        print("It might have been saved incorrectly or truncated.")
        return
    except EOFError:
        print("\n[Error] File is empty.")
        return
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred during loading: {e}")
        return

    # 3. Inspect the Loaded Data
    print(f"\nType of loaded data: {type(data)}")

    # Based on your script, it should be a dictionary
    if isinstance(data, dict):
        num_items = len(data)
        print(f"Data is a dictionary with {num_items} items (tokens).")
        
        if num_items == 0:
            print("The dictionary is empty.")
            return

        print("\n--- Sample of first 5 items ---")
        count = 0
        for key, value in data.items():
            print(f"\nItem {count + 1}:")
            print(f"  Key (token):   {repr(key)}") # repr() shows quotes around strings
            print(f"  Value Type:    {type(value)}")
            
            # If it's a numpy array, show details
            if isinstance(value, np.ndarray):
                print(f"  Value Shape:   {value.shape}")
                print(f"  Value Dtype:   {value.dtype}")
                # Print a small snippet of the vector
                print(f"  Value Snippet: [{value[0]}, {value[1]}, ...]")
            else:
                # If not an array, just print the value (truncated)
                print(f"  Value (snippet): {str(value)[:80]}...")
            
            count += 1
            if count >= 5:
                break
    
    # Handle other common data types just in case
    elif isinstance(data, (list, tuple)):
        print(f"Data is a {type(data).__name__} with {len(data)} items.")
        print("\n--- Sample of first 5 items ---")
        for i, item in enumerate(data[:5]):
            print(f"  Item {i}: {type(item)} -- {str(item)[:80]}...")
            
    else:
        # Handle other types (like a single object, number, or string)
        print("\nData is not a dictionary or list. Printing snippet:")
        print(str(data)[:200] + "...")


if __name__ == "__main__":
    # Check if the user has changed the default path
    if "event_embeddings.pkl" in FILE_PATH:
        print("="*60)
        print("!! ACTION REQUIRED !!")
        print("\nPlease open this Python script and edit the 'FILE_PATH' variable")
        print("to point to your 24 KB .pkl file.")
        print("="*60)
        sys.exit(1) # Exit with an error
        
    inspect_pickle_file(FILE_PATH)