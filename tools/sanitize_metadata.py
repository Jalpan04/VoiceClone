import pandas as pd
import os

# Paths
train_csv = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\dataset\metadata_train.csv"
eval_csv = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\dataset\metadata_eval.csv"

def sanitize_csv(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    df = pd.read_csv(file_path, sep="|", header=0)
    
    # Truncate text to 150 chars (XTTS limit)
    # We also strip trailing/leading spaces
    df['text'] = df['text'].str.strip().str[:150]
    
    # Drop rows where text became empty
    df = df[df['text'].str.len() > 0]
    
    df.to_csv(file_path, sep="|", index=False)
    print(f"Sanitized {file_path}: Truncated text to 150 chars.")

sanitize_csv(train_csv)
sanitize_csv(eval_csv)
