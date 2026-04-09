"""
Restore annotations.json to only the original manual annotations,
wiping all auto-tagged entries added by auto_annotator.py.

The script identifies the original manual annotations by key order:
the original file ended at a specific key, and everything after that
was added by the auto-annotator.

Run this script BEFORE re-running auto_annotator.py with adjusted thresholds.
"""

import json
import shutil
from pathlib import Path


ANNOTATIONS_FILE = "annotations.json"
BACKUP_FILE = "annotations.backup.json"

# The last key that existed in the original manually annotated file (514 entries).
# Everything at or before this key in insertion order is considered manual.
ORIGINAL_LAST_KEY = "output/Mummy 2026-01-09 20-16-23/SPEAKER_00"


def main():
    ann_path = Path(ANNOTATIONS_FILE)
    backup_path = Path(BACKUP_FILE)

    if not ann_path.exists():
        print(f"ERROR: {ANNOTATIONS_FILE} not found.")
        return

    # Create a backup first
    shutil.copy(ann_path, backup_path)
    print(f"Backed up current annotations to: {BACKUP_FILE}")

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    keys = list(ann.keys())
    total_before = len(keys)

    if ORIGINAL_LAST_KEY not in keys:
        print(f"ERROR: Could not find the original last key in annotations.json.")
        print(f"  Expected: {ORIGINAL_LAST_KEY}")
        print("  Cannot safely restore. Aborting.")
        return

    last_idx = keys.index(ORIGINAL_LAST_KEY)
    original_keys = set(keys[: last_idx + 1])

    restored = {k: v for k, v in ann.items() if k in original_keys}
    total_after = len(restored)
    removed = total_before - total_after

    print(f"Original entries found: {total_after}")
    print(f"Auto-tagged entries removed: {removed}")
    print(f"  Mine:     {sum(1 for v in restored.values() if v == 'mine')}")
    print(f"  Not Mine: {sum(1 for v in restored.values() if v == 'not_mine')}")

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(restored, f, indent=2, ensure_ascii=False)

    print(f"\nRestored {ANNOTATIONS_FILE} to {total_after} manual annotations.")
    print(f"Backup saved at {BACKUP_FILE} in case you need to undo this.")


if __name__ == "__main__":
    main()
