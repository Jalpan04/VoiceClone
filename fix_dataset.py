"""
fix_dataset.py - Sanitizes wav file names and rebuilds metadata CSVs for XTTS training.
Run with: python fix_dataset.py
"""

import os
import re
import shutil

WAVS_DIR = r"D:\python projects\clone\TTS_Dataset\wavs"
DATASET_DIR = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\dataset"
METADATA_ORIG = r"D:\python projects\clone\TTS_Dataset\metadata.csv"
TRAIN_CSV = os.path.join(DATASET_DIR, "metadata_train.csv")
EVAL_CSV = os.path.join(DATASET_DIR, "metadata_eval.csv")
LANG = "hi"

print("=" * 60)
print("STEP 1: Read original metadata.csv")
print("=" * 60)

with open(METADATA_ORIG, "r", encoding="utf-8") as f:
    raw_lines = [l.rstrip("\n") for l in f.readlines() if l.strip()]

print(f"  Found {len(raw_lines)} entries in metadata.csv")

# Parse: format is "filename.wav|text"
entries = []
skipped = 0
for line in raw_lines:
    if line.startswith("audio_file") or line.startswith("wavs/"):
        # skip header or already-prefixed lines from previous run
        pass
    parts = line.split("|", 1)
    if len(parts) < 2:
        skipped += 1
        continue
    old_name = parts[0].strip()
    text = parts[1].strip()
    # old_name might be bare filename or full path
    if os.sep in old_name or "/" in old_name:
        old_name = os.path.basename(old_name)
    entries.append((old_name, text))

print(f"  Parsed {len(entries)} entries, skipped {skipped}")

print("\n" + "=" * 60)
print("STEP 2: Rename wav files to safe names")
print("=" * 60)

# Build a map: old_name -> new_safe_name
rename_map = {}
existing = os.listdir(WAVS_DIR)
for i, (old_name, text) in enumerate(entries):
    new_name = f"s{i:05d}.wav"
    rename_map[old_name] = new_name

# Check which files actually exist on disk
not_found = []
renamed = 0
for old_name, new_name in rename_map.items():
    old_path = os.path.join(WAVS_DIR, old_name)
    new_path = os.path.join(WAVS_DIR, new_name)
    if os.path.exists(new_path):
        renamed += 1
        continue  # already renamed
    if not os.path.exists(old_path):
        not_found.append(old_name)
        continue
    try:
        os.rename(old_path, new_path)
        renamed += 1
    except Exception as e:
        print(f"  ERROR renaming {old_name}: {e}")

print(f"  Renamed/verified: {renamed}")
if not_found:
    print(f"  NOT FOUND on disk ({len(not_found)}):")
    for n in not_found[:10]:
        print(f"    {n}")

print("\n" + "=" * 60)
print("STEP 3: Verify renamed files exist")
print("=" * 60)

valid_entries = []
for old_name, text in entries:
    new_name = rename_map.get(old_name, old_name)
    abs_path = os.path.join(WAVS_DIR, new_name)
    if os.path.exists(abs_path):
        valid_entries.append((abs_path.replace("\\", "/"), text))
    else:
        pass  # silently skip missing

print(f"  Valid wav+text pairs: {len(valid_entries)}")

print("\n" + "=" * 60)
print("STEP 4: Split into train/eval and write CSVs")
print("=" * 60)

import random
random.seed(42)
random.shuffle(valid_entries)

num_eval = max(1, int(len(valid_entries) * 0.15))
eval_entries = valid_entries[:num_eval]
train_entries = valid_entries[num_eval:]
print(f"  Train: {len(train_entries)}, Eval: {len(eval_entries)}")

os.makedirs(DATASET_DIR, exist_ok=True)

def write_csv(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("audio_file|text|speaker_name\n")
        for audio, text in entries:
            safe_text = text.replace("|", " ")
            f.write(f"{audio}|{safe_text}|coqui\n")
    print(f"  Written: {path} ({len(entries)} rows)")

write_csv(TRAIN_CSV, train_entries)
write_csv(EVAL_CSV, eval_entries)

# Update lang.txt
with open(os.path.join(DATASET_DIR, "lang.txt"), "w") as f:
    f.write(LANG)
print(f"  lang.txt set to: {LANG}")

print("\n" + "=" * 60)
print("STEP 5: Remove stale 'run' folder (unlock)")
print("=" * 60)

run_dir = r"D:\python projects\clone\xtts-finetune-webui\finetune_models\run"
if os.path.exists(run_dir):
    try:
        shutil.rmtree(run_dir)
        print(f"  Deleted: {run_dir}")
    except Exception as e:
        print(f"  Could not delete run dir (may still be locked): {e}")
        print("  Please close the Gradio WebUI window and try again.")
else:
    print(f"  No 'run' folder found, nothing to clean.")

print("\n" + "=" * 60)
print("DONE. Check the output above for any errors.")
print("Sample train CSV rows:")
print("=" * 60)
with open(TRAIN_CSV, "r", encoding="utf-8") as f:
    for i, l in enumerate(f):
        print(f"  {l.rstrip()}")
        if i >= 3:
            break
