# Each subject has a randomized block order, with one block per condition.
# Each image appears exactly once per block, resulting in 6 repetitions per image.
# Specific masks are provided for each image for the "suppressive" "adversarial" and "phase_scrambled" conditions.
# For "Mondrian" and "Gaussian" conditions, masks are generated within the player (on the fly).
# The CSV file is ready for the player and contains all the necessary information.

from pathlib import Path
import random
import csv

# =========================
# PATHS
# =========================
CODE_ROOT = Path(__file__).parent
STIM_SET_DIR = CODE_ROOT / "stimulus_set"
MASK_SET_DIR = CODE_ROOT / "mask_set" 
DATA_DIR = CODE_ROOT / "data"

# =========================
# CONDITIONS & LABELS
# =========================
CONDITIONS = [
    "unmasked",
    "gaussian",
    "mondrian",
    "suppressive",
    "adversarial",
    "phase_scrambled"
]

STIM_LABELS = [
    "dog","cat","man","woman",
    "airplane","truck","cup","bag"
]

# =========================
# EXPERIMENT BUILDER
# =========================
class ExperimentBuilder:

    def __init__(self, subject_id):
        self.subject_id = subject_id
        self.rng = random.Random(subject_id)
        self.sub_dir = DATA_DIR / f"sub_{subject_id}"
        self.sub_dir.mkdir(parents=True, exist_ok=True)

    def get_all_images(self):
        """Retrieves all images with their label and full path"""
        images = []
        for label in STIM_LABELS:
            folder = STIM_SET_DIR / label.upper()
            for img in sorted(folder.glob("*.png")):
                images.append((label, img.name, str(img)))
        return images

    def randomize_block_order(self):
        order = CONDITIONS.copy()
        self.rng.shuffle(order)
        return order

    def build_trials(self):
        images = self.get_all_images()
        block_order = self.randomize_block_order()
        trials = []
        trial_global = 1

        for block_idx, cond in enumerate(block_order, start=1):
            block_images = images.copy()
            self.rng.shuffle(block_images)

            for trial_idx, (label, img_name, img_path) in enumerate(block_images, start=1):
                trials.append({
                    "subject_id": self.subject_id,
                    "block": block_idx,
                    "trial": trial_idx,
                    "trial_global": trial_global,
                    "stim_label": label,            
                    "stim_image": img_name,
                    "stim_path": img_path,
                    "mask_type": cond,
                    "mask_path": None,
                    "adversarial_label": ""
                })
                trial_global += 1

        return trials

    def assign_masks(self, trials):

        for t in trials:
            cond = t["mask_type"]
            label = t["stim_label"].upper()
            img_name_upper = t["stim_image"].split(".")[0].upper()

            if cond == "unmasked":
                t["mask_path"] = ""

            elif cond == "gaussian":
                t["mask_path"] = "gaussian"

            elif cond == "mondrian":
                t["mask_path"] = "mondrian"

            elif cond == "suppressive":
                t["mask_path"] = str(
                    MASK_SET_DIR /
                    "SUPPRESSIVE" /
                    f"suppressive_{label}" /
                    f"suppressive_{img_name_upper}.png"
                )

            elif cond == "adversarial":
                # image name without extension (e.g. BAG_01 → BAG01)
                img_base = t["stim_image"].split(".")[0].upper().replace("_", "")

                # folder containing adversarial masks
                adv_folder = (
                    MASK_SET_DIR /
                    "ADVERSARIAL" /
                    f"adversarial_{label}"
                )

                # find the corresponding file (pattern BAG01_TO_XXX.png)
                matches = list(adv_folder.glob(f"{img_base}_TO_*.png"))

                if len(matches) != 1:
                    raise ValueError(f"[ERROR] Expected 1 adversarial mask for {img_base}, found {len(matches)}")

                mask_path = matches[0]
                t["mask_path"] = str(mask_path)

                # EXTRACTION OF TARGET LABEL   e.g. BAG01_TO_CUP.png → CUP
                target = mask_path.stem.split("_TO_")[1]
                t["adversarial_label"] = target

            elif cond == "phase_scrambled":     # creates a variable using format expected by the player (- instead of _)
                cond_fixed = "phase-scrambled"  # corrects condition name for the player so that in the final CSV, `mask_type` column displays `phase-scrambled` rather than `phase_scrambled`
                t["mask_type"] = cond_fixed     # for the player to detect the file-based mask, constructs file path to locate the corresponding image file in the correct folder
                t["mask_path"] = str(
                    MASK_SET_DIR /
                    "PHASE-SCRAMBLED" /
                    f"phase_scrambled_{label}" /
                    f"ps_{img_name_upper}.png"
                )

    def save_csv(self, trials):
        out = self.sub_dir / f"sub_{self.subject_id}_trial_structure.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=trials[0].keys())
            writer.writeheader()
            writer.writerows(trials)
        return out

    def run(self):
        trials = self.build_trials()
        self.assign_masks(trials)
        path = self.save_csv(trials)
        print("saved", path)

# =========================
# RUN GENERATOR
# =========================
if __name__ == "__main__":
    for sub in range(1, 21):
        ExperimentBuilder(sub).run()