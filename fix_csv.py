import os

# --- IMPORTANT: Set this to the full path of your corrupt file ---
corrupt_file_path = r'C:\Users\DEVESH PALO\projects\GAIA-DataSet-main\GAIA-DataSet-main\GAIA-DataSet-main\MicroSS\business\business_table_webservice1_2021-07.csv'
# ------------------------------------------------------------------

print(f"Attempting to fix: {os.path.basename(corrupt_file_path)}")

try:
    # Get the size of the file before fixing
    original_size = os.path.getsize(corrupt_file_path)
    
    # Open the file in 'append' mode ('a') and add a closing quote and a newline
    with open(corrupt_file_path, 'a', encoding='utf-8') as f:
        # We add a newline first to ensure the quote is on a new line,
        # then the quote, then another newline for a clean end.
        f.write('\n"\n')

    # Get the new size
    new_size = os.path.getsize(corrupt_file_path)

    print("\n[SUCCESS]")
    print(f"  - Appended a closing quote to the end of the file.")
    print(f"  - Original size: {original_size} bytes")
    print(f"  - New size:      {new_size} bytes")
    
except FileNotFoundError:
    print(f"[ERROR] File not found. Please double-check the path:\n{corrupt_file_path}")
except Exception as e:
    print(f"[ERROR] An unexpected error occurred: {e}")