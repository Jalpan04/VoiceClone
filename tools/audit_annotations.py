"""
Audit the current state of annotations.json and autoaot.json.
Shows statistics, suspicious conversations, and per-contact breakdowns.
"""

import json
from collections import defaultdict
from pathlib import Path


ANNOTATIONS_FILE = "annotations.json"
AUTOAOT_FILE = "autoaot.json"


def load_json(path):
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_conversation(ann):
    conv_map = defaultdict(dict)
    for key, label in ann.items():
        parts = key.rsplit("/", 1)
        if len(parts) == 2:
            conv_map[parts[0]][parts[1]] = label
    return conv_map


def main():
    ann = load_json(ANNOTATIONS_FILE)
    auto = load_json(AUTOAOT_FILE)

    mine_count = sum(1 for v in ann.values() if v == "mine")
    not_mine_count = sum(1 for v in ann.values() if v == "not_mine")
    unsure_count = len(auto)

    print("=" * 60)
    print("ANNOTATION SUMMARY")
    print("=" * 60)
    print(f"  Mine:      {mine_count}")
    print(f"  Not Mine:  {not_mine_count}")
    print(f"  Unsure:    {unsure_count}")
    print(f"  Total:     {mine_count + not_mine_count + unsure_count}")
    print()

    # Group by conversation
    conv_map = group_by_conversation(ann)
    total_convs = len(conv_map)
    mine_convs = {c: s for c, s in conv_map.items() if "mine" in s.values()}
    no_mine_convs = {c: s for c, s in conv_map.items() if "mine" not in s.values()}

    print("=" * 60)
    print("CONVERSATION BREAKDOWN")
    print("=" * 60)
    print(f"  Total conversations:                      {total_convs}")
    print(f"  Conversations with at least one 'mine':   {len(mine_convs)}")
    print(f"  Conversations with zero 'mine' speakers:  {len(no_mine_convs)}")
    print()

    # Flag suspicious: 2-speaker calls with no mine (likely a real call where you spoke)
    two_speaker_no_mine = {c: s for c, s in no_mine_convs.items() if len(s) == 2}
    print("=" * 60)
    print(f"SUSPICIOUS: Two-speaker calls with ZERO mine speakers: {len(two_speaker_no_mine)}")
    print("(These are likely calls where the auto-tagger made a mistake)")
    print("=" * 60)
    for conv, spks in list(two_speaker_no_mine.items())[:20]:
        contact = conv.replace("output/", "").rsplit(" ", 2)[0]
        print(f"  [{contact}]  {conv.replace('output/', '')}")
    if len(two_speaker_no_mine) > 20:
        print(f"  ... and {len(two_speaker_no_mine) - 20} more.")
    print()

    # Per-contact breakdown
    print("=" * 60)
    print("PER-CONTACT MINE COUNT")
    print("=" * 60)
    contact_mine = defaultdict(int)
    contact_total = defaultdict(int)
    for conv, spks in conv_map.items():
        contact = conv.replace("output/", "").rsplit(" ", 2)[0]
        for label in spks.values():
            contact_total[contact] += 1
            if label == "mine":
                contact_mine[contact] += 1

    # Sort by mine count desc
    sorted_contacts = sorted(contact_mine.items(), key=lambda x: -x[1])
    for contact, mine in sorted_contacts[:30]:
        total = contact_total[contact]
        print(f"  {contact:<35} mine: {mine:>3} / {total:>3} speakers")

    if not sorted_contacts:
        print("  No 'mine' annotations found yet.")

    # Autoaot score distribution
    if auto:
        scores = [v["similarity"] for v in auto.values()]
        print()
        print("=" * 60)
        print(f"UNSURE (autoaot.json) SIMILARITY DISTRIBUTION  [{len(scores)} items]")
        print("=" * 60)
        print(f"  Min:  {min(scores):.3f}")
        print(f"  Max:  {max(scores):.3f}")
        print(f"  Avg:  {sum(scores)/len(scores):.3f}")

        buckets = {"0.65-0.70": 0, "0.70-0.75": 0, "0.75-0.80": 0, "0.80-0.85": 0}
        for s in scores:
            if s < 0.70:
                buckets["0.65-0.70"] += 1
            elif s < 0.75:
                buckets["0.70-0.75"] += 1
            elif s < 0.80:
                buckets["0.75-0.80"] += 1
            else:
                buckets["0.80-0.85"] += 1
        for bucket, count in buckets.items():
            print(f"  {bucket}:  {count}")


if __name__ == "__main__":
    main()
