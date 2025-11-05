import os

embedding_script = r'C:\Users\DEVESH PALO\projects\DiagFusionWorking\diagf\tools\create_sentence_embedding.py'

print("="*80)
print("FULL EMBEDDING CREATION SCRIPT")
print("="*80)

with open(embedding_script, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"\nTotal lines: {len(lines)}\n")

# Print the entire script with line numbers
for i, line in enumerate(lines, 1):
    print(f"{i:4d} | {line.rstrip()}")