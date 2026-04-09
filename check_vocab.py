import json
import os

vocab_path = r"D:\python projects\clone\xtts-finetune-webui\base_models\v2.0.2\vocab.json"

if os.path.exists(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    print(f"Vocab size: {len(vocab)}")
    
    # Check for Hindi characters as a sanity check
    hindi_chars = [c for c in vocab.keys() if '\u0900' <= c <= '\u097f']
    print(f"Hindi characters found: {len(hindi_chars)}")
    
    if len(vocab) != 6153:
        print(f"MISMATCH! Found {len(vocab)}, expected 6153.")
else:
    print("Vocab file not found.")
