## This script provides functions for loading and processing an image dataset stored in HDF5 format.
## The code is designed to be used with PyTorch and torchvision for data loading and transformations.

## It includes:
##    - H5Dataset: A PyTorch Dataset class for loading images and labels from an HDF5 file.
##    - get_image_indices: A function to sample dataset indices by class, either by specific mapping or random sampling.
##    - custom_collate_fn: A function to create a collate function for building temporal image sequences with various options, including noise and masks

import os
import re
import h5py
import random
import numpy as np
from typing import Callable, Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


######################
# HDF5 DATASET CLASS
######################

class H5Dataset(Dataset):      
    """
    PyTorch Dataset for loading images and labels from an HDF5 file.

    Each split (e.g., 'train', 'val', 'test') is expected to contain two
    datasets: 'data' and 'labels'.

    Args:
        h5_path (str): Path to the HDF5 file containing the dataset
        split (str): Data split to load ('train', 'val', 'test', etc.)
        transform (Optional[Callable], optional): Transform function to apply to images.
            If None, images are converted to tensors using ToTensor(). Defaults to None.
    
    Raises:
        FileNotFoundError: If the HDF5 file doesn't exist
        KeyError: If the specified split doesn't exist in the file
    """


    def __init__(
        self,
        h5_path: str,
        split: str,
        transform: Optional[Callable] = None
    ):
        super().__init__()
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as h5_file:
            if split not in h5_file:
                raise KeyError(
                    f"Split '{split}' not found in H5. Available: {list(h5_file.keys())}"
                )
            grp = h5_file[split]
            self.data = grp['data'][:]    # shape: (N, C, H, W)  
            self.labels = grp['labels'][:]  # shape: (N,)

        self.transform = transform

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.data)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Retrieve image and label at index `idx`.

        Args:
            idx (int): Index of the sample to retrieve (0 <= idx < len(dataset))
        
        Returns:
            image: Tensor of shape (C, H, W).
            label: Integer class label.
        Raises:
            IndexError: If idx is out of bounds
        """
        img = self.data[idx]
        lbl = int(self.labels[idx])

        if self.transform:
            img = self.transform(img)
        else:
            # Default to converting numpy array to tensor
            img = transforms.ToTensor()(img)

        return img, lbl
    

#########################
# SAMPLE DATASET INDICES
#########################

def get_image_indices(    
    dataset,
    mode: str = 'specific',          
    class_to_indices: dict = None,   
    num_classes: int = 100,          
    images_per_class: int = 1,       
    seed: int = 42
):
    """
    Sample dataset indices either by specific mapping or by random sampling per class.
    
    This function provides two modes for selecting dataset indices:
    1. 'specific': Use a predefined mapping of classes to indices
    2. 'random': Randomly sample a fixed number of images per class
    
    Args:
        dataset (Dataset): PyTorch dataset with __len__ and __getitem__ methods
        mode (str, optional): Sampling mode - 'specific' or 'random'. Defaults to 'specific'.
        class_to_indices (Optional[Dict[int, List[int]]], optional): 
            Mapping from class labels to lists of dataset indices. 
            Required when mode='specific'. Defaults to None.
        num_classes (int, optional): Total number of classes in dataset. 
            Used when mode='random'. Defaults to 100.
        images_per_class (int, optional): Number of images to sample per class
            when mode='random'. Defaults to 1.
        seed (int, optional): Random seed for reproducible sampling. Defaults to 42.
    
    Returns:
        List[int]: selected indices.
        Dict[int,int]: mapping index -> class label.

    Raises:
        ValueError: If mode is not 'specific' or 'random'
        ValueError: If mode='specific' but class_to_indices is None
    """
    # Set random seed for reproducibility
    random.seed(seed)

    index_to_class = {}      
    selected = []        

    if mode == 'specific':
        # Use predefined class-to-indices mapping
        if class_to_indices is None:
            raise ValueError("class_to_indices must be provided for specific mode")
        for cls, idxs in class_to_indices.items():
            for idx in idxs:
                selected.append(idx)
                index_to_class[idx] = cls

    elif mode == 'random':
        # Random sampling: collect indices for each class
        class_indices = {c: [] for c in range(num_classes)}
        # scan dataset once to fill class lists
        for i in range(len(dataset)):
            _, label = dataset[i]
            lbl = label.item() if torch.is_tensor(label) else label
            if len(class_indices[lbl]) < images_per_class:
                class_indices[lbl].append(i)
            # stop early if all filled
            if all(len(v) >= images_per_class for v in class_indices.values()):
                break
        for cls, idxs in class_indices.items():
            for idx in idxs:
                selected.append(idx)
                index_to_class[idx] = cls
    else:
        raise ValueError("mode must be 'specific' or 'random'")

    return selected, index_to_class



###############################
# CUSTOM DATA LOADER FUNCTION
###############################


def custom_collate_fn(
    option: str,
    n_timesteps: int,
    noise_std: float,
    mask_folder: Optional[str] = None
):
    """
    Factory function to create specialized collate functions for temporal image sequences.
    
    This function returns a collate_fn that transforms batches of individual images
    into temporal sequences using various strategies. Useful for video analysis,
    temporal modeling, and sequence-based deep learning tasks.
    
    Temporal Sequence Options:
        - 'same': Repeat the same image across all timesteps
        - 'zero': First frame is the image, subsequent frames are zeros  
        - 'noise': First frame is the image, subsequent frames are Gaussian noise
        - 'first_half': Image repeated for first half of timesteps, zeros for second half
        - 'first_two_then_noise': Image for first two timesteps, noise for remainder
        - 'image_zero_noise': First=image, second=zeros, third=noise (requires n_timesteps≥3)
        - 'image_noise_noise': First=image, second&third=same noise (requires n_timesteps≥3)    # Creates suppressive and adversarial masks
        - 'image_zero_mask': First=image, second=zeros, third=mask (requires mask_folder)
        - 'image_mask_mask': First=image, second&third=same mask (requires mask_folder)         # Charges the created mask for the test loop
    
    Args:
        option (str): Temporal sequence generation strategy (see options above)
        n_timesteps (int): Number of timesteps in each sequence. Must be ≥ 1.
        noise_std (float): Standard deviation for Gaussian noise generation. 
        mask_folder (Optional[str], optional): Path to folder containing mask files
            named 'bag_mask_{idx}.npy'. Required for mask-based options. Defaults to None.

    Returns:
        collate_fn(batch) -> (sequences, labels).
    
    Raises:
        ValueError: If option is not recognized
        ValueError: If mask_folder is required but not provided or invalid
        FileNotFoundError: If mask_folder doesn't exist
    """
    # Preload mask paths for mask-based options
    mask_paths: List[str] = []
    if mask_folder:
        # Ensure the folder exists
        if not os.path.isdir(mask_folder):
            raise FileNotFoundError(f"mask_folder '{mask_folder}' does not exist")
        # Regex to capture your mask naming scheme
        pattern = re.compile(r"image_batch(\d+)_img(\d+)_mask_opt_(\w+)\.npy")
        # Loop over all files in mask_folder so `fname` is defined
        for fname in os.listdir(mask_folder):
            m = pattern.match(fname)
            if m:
                mask_paths.append(os.path.join(mask_folder, fname))
        mask_paths.sort()  # deterministic ordering

    to_tensor = transforms.ToTensor()
    mask_counter = 0

    def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal collate function that processes a batch of samples.
        
        Args:
            batch: List of (image_tensor, label) tuples from the dataset
            
        Returns:
            Tuple of (sequences, labels) where:
            - sequences: torch.Tensor of shape (batch_size, n_timesteps, channels, height, width)
            - labels: torch.Tensor of shape (batch_size,) containing class labels
        """
        nonlocal mask_counter
        B = len(batch)
        C, H, W = batch[0][0].shape

        # Initialize output tensors
        
        sequences = torch.zeros((B, n_timesteps, C, H, W))
        labels = []

        for i, (img, lbl) in enumerate(batch):
            # Set first timestep to the original image for all options
            sequences[i, 0] = img
            labels.append(lbl)

            # Generate subsequent timesteps based on the chosen option
            if option == 'same':
                sequences[i] = img.unsqueeze(0).repeat(n_timesteps, 1, 1, 1)
            elif option == 'zero':                                                 
                pass                                                       
            elif option == 'noise':
                sequences[i, 1:] = torch.randn((n_timesteps - 1, C, H, W)) * noise_std
            elif option == 'first_half':
                half = n_timesteps // 2
                sequences[i, :half] = img.unsqueeze(0).repeat(half, 1, 1, 1) 
            elif option == 'first_two_then_noise':
                sequences[i, :2] = img.unsqueeze(0).repeat(2, 1, 1, 1)
                sequences[i, 2:] = torch.randn((n_timesteps - 2, C, H, W)) * noise_std
            elif option == 'image_zero_noise':
                sequences[i, 1] = 0
                sequences[i, 2] = 0.5 + torch.randn((C, H, W)) * noise_std 
            elif option == 'image_noise_noise':                                 # same noise at t1 et t2
                noise = 0.5 + torch.randn((C, H, W)) * noise_std
                sequences[i, 1] = noise
                sequences[i, 2] = noise
            elif option in ('image_zero_mask', 'image_mask_mask'): 
                if not mask_folder or mask_counter >= len(mask_paths):  
                    raise ValueError("Mask files not available for mask-based option")
                mask_np = np.load(mask_paths[mask_counter])
                mask_t = to_tensor(mask_np)
                if option == 'image_zero_mask':
                    sequences[i, 1] = 0
                    sequences[i, 2] = mask_t
                else:  # image_mask_mask 
                    sequences[i, 1] = mask_t
                    sequences[i, 2] = mask_t
                mask_counter += 1
            else:
                raise ValueError(f"Unknown option: {option}")

        return sequences, torch.tensor(labels)

    return collate_fn 



