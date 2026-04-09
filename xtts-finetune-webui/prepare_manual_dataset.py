import os
import shutil
from pathlib import Path

# Paths
source_csv = r"D:\python projects\clone\TTS_Dataset\metadata.csv"
output_dir = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\dataset"

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Read metadata
try:
    with open(source_csv, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Split (95% train, 5% eval)
    split_idx = int(len(lines) * 0.95)
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]

    # Write files
    with open(os.path.join(output_dir, "metadata_train.csv"), 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(os.path.join(output_dir, "metadata_eval.csv"), 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)

    # Write language file
    with open(os.path.join(output_dir, "lang.txt"), 'w', encoding='utf-8') as f:
        f.write("hi")

    print(f"Successfully prepared {len(train_lines)} training lines and {len(eval_lines)} eval lines.")
except Exception as e:
    print(f"Error: {e}")
