## This script tests the model on a subset of 100 images from the test set, ensuring each class is represented.

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm.notebook import tqdm
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from dataset_functions_new import H5Dataset, custom_collate_fn
from helper_functions_try import test
from model_functions_new import BLT_net_64


# 1) Load H5Dataset with ToTensor and Normalize transforms

transform_list = [transforms.ToTensor()]

with h5py.File('./miniecoset_64px.h5', 'r') as h5_file:
    imchmean = h5_file['train_img_mean_channels'][:]
    imchstd = h5_file['train_img_std_channels'][:]

transform_list.append(transforms.Normalize(mean=imchmean/255, std=imchstd/255))
transform = transforms.Compose(transform_list)

minieco_test = H5Dataset("./miniecoset_64px.h5", split="test", transform=transform)

# 2) Select 100 images from the test set, ensuring each class is represented
# We will select 1 image per class, ensuring we cover all classes in the dataset.

num_classes = 100
class_indices = {cls: None for cls in range(num_classes)}

for idx in range(len(minieco_test)):
    img, label = minieco_test[idx]
    if torch.is_tensor(label):
        label = label.item()
    if class_indices[label] is None:
        class_indices[label] = idx
    if all(v is not None for v in class_indices.values()):
        break

subset_1 = sorted(list(class_indices.values()))[:25]
print(f"Selected 100 indices: {subset_1}")

subset_dataset = Subset(minieco_test, subset_1)

# 3) Load the model
net = BLT_net_64(timesteps=10)
sush_savedir = './b64_l_1_t_1_ltinteract_additive_T_10_ltposition_all_num_1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.load_state_dict(torch.load(f'{sush_savedir}/b64_l_1_t_1_ltinteract_additive_T_10_ltposition_all_num_1_epoch_59.pth', map_location=device, weights_only=True))
net = net.to(device)

# 4) Create DataLoader with custom collate function

n_timesteps = 10
batch_size = 16
collate_fn = custom_collate_fn(option="image_mask_mask", n_timesteps=n_timesteps, noise_std=0.1, mask_folder = "./xxx_to_class5_with_lossmetrics_NEU/np_masks")

dataloader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# 5) Test the model on the selected subset

criterion = nn.CrossEntropyLoss()

test_loss, accuracy = test(net, dataloader, criterion, device, save_dir='./TEST23JULI', save=True)

print(f"✅ Final Test Results - Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")
