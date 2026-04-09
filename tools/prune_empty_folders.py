"""
Prune empty speaker folders from annotations.json and autoaot.json.
Removes any entry where the corresponding folder has no .wav audio files.
Also removes the empty folders from disk.
"""

import json
import os
import shutil
from pathlib import Path


ANNOTATIONS_FILE = "annotations.json"
AUTOAOT_FILE = "autoaot.json"


def has_audio(folder_path: str) -> bool:
    p = Path(folder_path)
    if not p.exists():
        return False
    return any(p.glob("*.wav"))


def main():
    ann_path = Path(ANNOTATIONS_FILE)
    auto_path = Path(AUTOAOT_FILE)

    ann = {}
    auto = {}

    if ann_path.exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

    if auto_path.exists():
        with open(auto_path, "r", encoding="utf-8") as f:
            auto = json.load(f)

    # --- Prune annotations.json ---
    clean_ann = {}
    removed_ann = []
    for key, label in ann.items():
        if has_audio(key):
            clean_ann[key] = label
        else:
            removed_ann.append(key)

    # --- Prune autoaot.json ---
    clean_auto = {}
    removed_auto = []
    for key, info in auto.items():
        if has_audio(key):
            clean_auto[key] = info
        else:
            removed_auto.append(key)

    # --- Remove empty folders from disk ---
    deleted_folders = 0
    all_removed = set(removed_ann + removed_auto)
    for folder_str in all_removed:
        p = Path(folder_str)
        if p.exists() and not any(p.iterdir()):
            shutil.rmtree(p)
            deleted_folders += 1

    # --- Save cleaned files ---
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(clean_ann, f, indent=2, ensure_ascii=False)

    with open(auto_path, "w", encoding="utf-8") as f:
        json.dump(clean_auto, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print("PRUNE COMPLETE")
    print("=" * 50)
    print(f"  Removed from annotations.json:  {len(removed_ann)}")
    print(f"  Removed from autoaot.json:      {len(removed_auto)}")
    print(f"  Empty folders deleted from disk: {deleted_folders}")
    print(f"  Remaining in annotations.json:  {len(clean_ann)}")
    print(f"  Remaining in autoaot.json:      {len(clean_auto)}")

    if removed_ann or removed_auto:
        print()
        print("Removed entries:")
        for r in removed_ann + removed_auto:
            print(f"  {r}")


if __name__ == "__main__":
    main()
