# Iterates through the CSV for a given subject. For each image_name:
# Counts how many times it appears (image_counts)
# Determines which mask_types (i.e., blocks) it appears in (image_blocks)
# Verifies that each image appears exactly 6 times and is present in all 6 blocks (exactly once per block)
# Creates a 'counterbalancing_check' CSV that counts the occurrences for each image x mask combination (must equal 1).

from pathlib import Path
import csv
from collections import defaultdict, Counter

# =========================
# PARAMS
# =========================
DATA_DIR = Path("data")
SUBJECTS = range(1, 21)
BLOCKS = [
    'adversarial',
    'unmasked',
    'gaussian',
    'mondrian',
    'phase-scrambled',
    'suppressive'
]

# =========================
# CHECK 1 : BLOCK COUNTERBALANCING
# =========================
def check_counterbalancing(sub_id):
    csv_path = DATA_DIR / f"sub_{sub_id}" / f"sub_{sub_id}_trial_structure.csv"
    print(f"\nChecking subject {sub_id} -> {csv_path.name}")

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    image_blocks = defaultdict(set)
    image_counts = defaultdict(int)

    for row in rows:
        img = row['stim_image']
        blk = row['mask_type']
        image_blocks[img].add(blk)
        image_counts[img] += 1

    all_ok = True
    for img in sorted(image_counts.keys()):
        count = image_counts[img]
        blocks = image_blocks[img]
        if count != 6:
            print(f"⚠️  {img} appears {count} times (should be 6)")
            all_ok = False
        if len(blocks) != 6:
            missing = set(BLOCKS) - blocks
            print(f"⚠️  {img} missing in blocks {missing}")
            all_ok = False

    if all_ok:
        print("✅ All images correctly counterbalanced (1 per block, 6 in total).")
    return all_ok


# =========================
# CHECK 2 : IMAGE × MASK PAIRING
# =========================
def check_image_mask_pairs(sub_id):

    csv_path = DATA_DIR / f"sub_{sub_id}" / f"sub_{sub_id}_trial_structure.csv"
    output_path = DATA_DIR / f"sub_{sub_id}" / f"sub_{sub_id}_counterbalancing_check.csv"

    print(f"\n[PAIR CHECK] Subject {sub_id} -> {csv_path.name}")

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)

        combos = [
            (row['stim_image'], row['mask_type'], row['stim_label'])
            for row in reader
        ]

    counts = Counter(combos)

    # =========================
    # WRITE CSV
    # =========================
    with open(output_path, 'w', newline='') as f_out:
        writer = csv.writer(f_out)

        writer.writerow(['stim_image', 'mask_type', 'stim_label', 'count'])

        for (stim_image, mask_type, label), cnt in sorted(counts.items()):
            writer.writerow([stim_image, mask_type, label, cnt])

    # =========================
    # COMBINATIONS CHECK
    # =========================
    ok = True

    for combo, cnt in counts.items():
        if cnt != 1:
            print(f"⚠️  Repetition detected: {combo} -> {cnt} times")
            ok = False
    if ok:
        print(f"✅ All combinations correct. CSV saved -> {output_path.name}") # checks if all image × mask combinations appears exactly once
    return ok

# =========================
# RUN FOR ALL SUBJECTS
# =========================
if __name__ == "__main__":
    for sub in SUBJECTS:
        check_counterbalancing(sub)
        check_image_mask_pairs(sub)