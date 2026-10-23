## This script optimizes images from a dataset using a neural network model.
## It is designed to work with a specific dataset format and model architecture, and includes functionality for
## handling different optimization methods and saving results in a structured manner.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import pickle
import random
import numpy as np
import h5py

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from collections import Counter
import pandas as pd

from model_functions_new import BLT_net_64
from dataset_functions_new import H5Dataset, custom_collate_fn
from helper_functions_try_ME import optimize_image, save_final_histograms


# 1) Load mapping
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = "/scratch/jcombe/supermasks/scripts/optimitze_mask_loop_new_ME_TEST_PER_CLASS/full_map.pkl"

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

full_map = data["full_map"]
idxs = data["idxs"]
index_to_class = data["index_to_class"]

idx_to_target = {s: t for s, _, t, _ in full_map}
idx_to_target_class = {s: tc for s, _, _, tc in full_map}

print("Mapping loaded:", len(full_map))

# 2) Slurm chunking for parralel processing on CURTA
job_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
num_jobs = 8                                        # to adapt according to the total number of jobs in the array

chunk_size = math.ceil(len(idxs) / num_jobs)

start = job_id * chunk_size
end = min(start + chunk_size, len(idxs))

idxs_chunk = idxs[start:end]                       # each job will process a chunk of indices

print(f"Job {job_id}/{num_jobs}")
print(f"Processing indices {start} → {end} (n={len(idxs_chunk)})")

# 3) Load H5Dataset with ToTensor and Normalize transforms
h5_path = '/scratch/jcombe/supermasks/miniecoset_64px.h5'   # Modify this path depending on where the H5 file is located on your system or cluster

transform_list = [transforms.ToTensor()]

with h5py.File(h5_path, 'r') as h5_file:
    imchmean = h5_file['train_img_mean_channels'][:]
    imchstd = h5_file['train_img_std_channels'][:]

transform_list.append(transforms.Normalize(mean=imchmean/255.0, std=imchstd/255.0))
transform = transforms.Compose(transform_list)

dataset = H5Dataset(
    h5_path,
    split="train",
    transform=transform
)

subset = Subset(dataset, idxs_chunk)

# 4) Define parameters
batch_size = 1
n_timesteps = 10
option = "image_noise_noise"
noise_std = 0.1

# 5) Create DataLoader for the subset
dataloader = DataLoader(
    subset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=custom_collate_fn(option, n_timesteps, noise_std)
)

# 6) Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 7) Load model and pretrained weights 
net = BLT_net_64(timesteps=5)
weights_path = "/scratch/jcombe/supermasks/b64_weights/b64_l_1_t_1_ltinteract_additive_T_10_ltposition_all_num_1_epoch_59.pth"

net.load_state_dict(
    torch.load(weights_path, map_location=device, weights_only=True)
)

# 8) Move model to device
net = net.to(device)
net.eval()

# 9) Set seeds for reproducibility 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 10) Optimization hyperparameters
iterations = 15000
learning_rate = 0.001
target_timestep = 3

# 11) Create directories to save results
save_dir = os.path.join(
    "/scratch/jcombe/supermasks/results/TEST_PER_CLASS_ADVERSARIAL",
    f"job_{job_id}"
)

os.makedirs(save_dir, exist_ok=True)

# Create per-class subfolders for organized outputs
for cls in set(index_to_class.values()):
    os.makedirs(os.path.join(save_dir, f"class_{cls}_optimisation"), exist_ok=True)

# 12) Sanity checks and CSV generation 
print("\nSanity check mapping size:", len(idx_to_target))
print("Sanity check idxs size:", len(idxs))

missing = [idx for idx in idxs if idx not in idx_to_target]
print("Missing keys:", len(missing))

same_image = sum([s == t for s, _, t, _ in full_map])   # Same-image assignments
print("\nNumber of same-image assignments:", same_image)
if same_image == 0:
    print("No self-target assignment OK")
else:
    print("⚠️ ERROR: self-target exists")

df = pd.DataFrame(full_map, columns=[   # CSV    
    "source_idx",  # = original_idx
    "source_class",
    "target_idx",
    "target_class"
])

df["same_image"] = df["source_idx"] == df["target_idx"]                  # Self-check

target_idx_counts = df["target_idx"].value_counts()                      # Target idx counts
df["target_idx_count"] = df["target_idx"].map(target_idx_counts)

target_class_counts = df["target_class"].value_counts()                  # Target class counts
df["target_class_count"] = df["target_class"].map(target_class_counts)

df["target_idx_unique_OK"] = df["target_idx_count"] == 1                 # Validations
df["target_class_balance_OK"] = df["target_class_count"] == 40

# 13) Save CSV (important: per job OR global)
csv_path = os.path.join(save_dir, f"target_mapping_check_job{job_id}.csv") # to run multiple jobs in parallel on CURTA; otherwise each job will write on the same file risking data to be overwritten
df.to_csv(csv_path, index=False)
print("\nCSV saved:", csv_path)
print("\nTarget_class distribution (should all be 40):")
print(target_class_counts.sort_index().to_string(index=True, header=False))

# Optimization loop
def optimize_loop():
    """
    Iterate through the DataLoader, optimize each image's mask sequence,
    collect initial/final activations, and save aggregated results.
    """
    initial_activations_all = []
    final_activations_all = []
    labels_all = []

    print("Starting optimization loop...")

    # Loop over each batch  for clear logging
    for batch_idx, (img_batch, label_batch) in enumerate(dataloader):
        # Move batch to device
        img_batch = img_batch.to(device)

        for i in range(img_batch.shape[0]):  # Expect batch_size=1 here

            # Dataloader -> original dataset alignment:
            # batch_idx is the position within the chunk (valid because batch_size=1 and shuffle=False), so idxs_chunk[batch_idx] gives the index of the image in the full source dataset
            source_idx = idxs_chunk[batch_idx]

            # Current image, keeping the batch dimension (shape [1, C, H, W])
            img = img_batch[i:i+1]

            # Target associated with this source image (index and class), used as the optimisation objective
            target_idx = idx_to_target[source_idx]
            target_class = idx_to_target_class[source_idx]
            # Original class of the source image
            img_class = index_to_class[source_idx]

            # Output directory, one per source class
            class_dir = os.path.join(save_dir, f"class_{img_class}_optimisation")

            print(
                f"[JOB {job_id}] Processing image {batch_idx + 1}/{len(dataloader)} | "
                f"source_idx = {source_idx} | "
                f"source_class = {img_class} | "
                f"target_idx = {target_idx} | "
                f"target_class = {target_class}"
            )

            # Run optimization
            optimized_seq, loss_hist, init_act, final_act, analysis = optimize_image(
                image=img,
                net=net,
                target_timestep=target_timestep,
                opt_method='target_random_class',
                iterations=iterations,
                batch_idx=batch_idx,
                image_idx=i,
                lr=learning_rate,
                device=device,
                save_dir=class_dir,
                class_label=img_class,
                target_class=target_class,
                save=True
            )

            # Collect activations and labels
            initial_activations_all.append(init_act.cpu())
            final_activations_all.append(final_act.cpu())
            labels_all.append(label_batch[i].item())

    # Save collected activations and labels as tensors
    if initial_activations_all:

        torch.save({
            'initial_activations': torch.stack(initial_activations_all),
            'final_activations': torch.stack(final_activations_all),
            'labels': torch.tensor(labels_all)
        }, os.path.join(save_dir, 'activation_data.pt'))

        # Save histograms for visual analysis
        save_final_histograms(save_dir, opt_method=None)
        print(f"[JOB {job_id}] Saved results.")
    else:
        print(f"[JOB {job_id}] No activations collected.")

# 14) Main Entry Point
if __name__ == "__main__":
    optimize_loop()
    print(f"Job {job_id} completed.")