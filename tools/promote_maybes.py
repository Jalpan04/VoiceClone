"""
Promote all 'maybe' items from autoaot.json with similarity >= threshold to 'mine'.
Items below threshold are moved into annotations.json as 'not_mine'.
Run this BEFORE launching the annotator UI so it can focus on re-checking not_mines.
"""

import json
from pathlib import Path

ANNOTATIONS_FILE = "annotations.json"
AUTOAOT_FILE = "autoaot.json"
THRESHOLD = 0.60


def main():
    ann_path = Path(ANNOTATIONS_FILE)
    auto_path = Path(AUTOAOT_FILE)

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    with open(auto_path, "r", encoding="utf-8") as f:
        auto = json.load(f)

    promoted_mine = []
    promoted_not_mine = []

    for folder, info in auto.items():
        sim = info.get("similarity", 0)
        if sim >= THRESHOLD:
            ann[folder] = "mine"
            promoted_mine.append((folder, sim))
        else:
            ann[folder] = "not_mine"
            promoted_not_mine.append((folder, sim))

    # Clear autoaot — everything has been resolved
    with open(auto_path, "w", encoding="utf-8") as f:
        json.dump({}, f, indent=2)

    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(ann, f, indent=2, ensure_ascii=False)

    print("=" * 55)
    print(f"PROMOTE MAYBES (threshold >= {int(THRESHOLD*100)}%)")
    print("=" * 55)
    print(f"  Promoted to MINE:      {len(promoted_mine)}")
    for folder, sim in promoted_mine:
        print(f"    [{sim:.0%}] {folder}")

    print(f"\n  Moved to NOT MINE:     {len(promoted_not_mine)}")
    for folder, sim in promoted_not_mine:
        print(f"    [{sim:.0%}] {folder}")

    mine_total = sum(1 for v in ann.values() if v == "mine")
    not_mine_total = sum(1 for v in ann.values() if v == "not_mine")
    print()
    print("=" * 55)
    print(f"  Total Mine:     {mine_total}")
    print(f"  Total Not Mine: {not_mine_total}")
    print(f"  autoaot.json:   cleared (0 remaining)")
    print()
    print("Now restart the annotator. The UI will queue all not_mine")
    print("items for your manual re-check.")


if __name__ == "__main__":
    main()
