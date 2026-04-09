import os
from huggingface_hub import hf_hub_download
import json

BASE_DIR = r"D:\python projects\clone\xtts-finetune-webui\base_models\v2.0.2"
os.makedirs(BASE_DIR, exist_ok=True)

REPO_ID = "coqui/XTTS-v2"
FILES = ["config.json", "model.pth"]

def sync():
    for filename in FILES:
        print(f"\n--- Downloading {filename} from {REPO_ID} ---")
        dest = os.path.join(BASE_DIR, filename)
        
        # Download using hf_hub_download which handles retries and resumable downloads
        file_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=BASE_DIR,
            local_dir_use_symlinks=False
        )
        print(f"  Saved to: {file_path}")

    print("\n\n=== All downloads complete ===")
    print("Verifying config.json...")
    
    config_path = os.path.join(BASE_DIR, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    token_count = cfg.get("model_args", {}).get("gpt_number_text_tokens", "NOT FOUND")
    print(f"gpt_number_text_tokens = {token_count}")

    if token_count == 6681:
        print("PASS: Token count matches vocab.json (6681). Training should now work.")
    else:
        print(f"WARNING: Expected 6681 but got {token_count}. Something may still be wrong.")

    model_size = os.path.getsize(os.path.join(BASE_DIR, "model.pth")) / 1024 / 1024
    print(f"model.pth size: {model_size:.1f} MB")

if __name__ == "__main__":
    sync()
