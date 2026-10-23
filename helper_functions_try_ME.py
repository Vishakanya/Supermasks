## This script contains various utility functions for testing, analyzing activations,
## calculating target loss, optimizing images, and sampling images based on luminance filtering.

## 1. Testing / Evaluation (`test` function):
##   - Evaluate a trained model using a test DataLoader
##   - Calculate overall loss and accuracy
##   - Save per-image losses and aggregate metrics to files

## 2. Activation Analysis (`analyze_activations`, `save_final_histograms`):
##   - Calculate norms and cosine similarity between activations before and after optimization
##   - Aggregate and visualize the distribution of cosine similarities

## 3. Target Loss Calculation (`compute_loss`):
##   - Dispatch to various adversarial or analytical loss objectives

## 4. Image Optimization (`optimize_image`):
##   - Optimize a dynamic sequence of masks to elicit a specific internal response
##     from the network
##   - Track and save the loss history, final activations,
##     optimized masks, and metrics

## 5. Dataset Sampling by Luminance (`sample_images_by_luminance`):
##   - Calculate the luminance of each image within a dataset partition
##   - Filter for images falling within ±σ of the global luminance distribution
   

import os
import csv
import random
from collections import defaultdict            
from typing import Tuple, List, Dict, Optional 

import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset  
from torchvision import transforms              
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset_functions_new import custom_collate_fn, H5Dataset    # La fonction custom_collate_fn construit des séquences temporelles (T,C,H,W) à partir d’une liste d'images (C,H,W) en injectant du bruit contrôlé, des répétitions ou variations (selon options je crois) pour stimuler la dynamique interne du modèle récurrent

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())

###############
# FUNCTION FOR SAFE TIMESTEP INDEXING
###############
def select_timestep(tensor: torch.Tensor, timestep: int) -> torch.Tensor:
    """
    Selects the timestep of a tensor.
    Handles 1D, 2D, and 3D+ tensors.
    """
    if tensor.dim() == 1:
        return tensor
    elif tensor.dim() == 2:
        # [B, T] or [B, features], take the timestep if possible
        if tensor.size(1) > timestep:
            return tensor[:, timestep]
        else:
            return tensor  # fallback: timestep out of range, return tensor unchanged
    elif tensor.dim() >= 3:
        slices = [slice(None), timestep] + [slice(None)]*(tensor.dim()-2)
        return tensor[tuple(slices)]
    else:
        raise ValueError(f"Unexpected tensor shape {tensor.shape}")
    
###############
# TEST FUNCTION
###############

def test(model, loader, criterion, device, save_dir='./results', save=True):
    """
    Evaluate a classification model on a test DataLoader.

    Args:
        model: Trained PyTorch model returning (logits, representations).
        loader: DataLoader for test data batches.
        criterion: Loss function (e.g., nn.CrossEntropyLoss()).
        device: Device (CPU/GPU) for computation.
        save_dir: Directory to save results files.
        save: Whether to write loss/accuracy and per-image CSV.

    Returns:
        avg_test_loss: Mean loss over all test samples.
        accuracy: Classification accuracy in percent.
    """   
    # Switch model to evaluation mode
    model.eval()

    test_loss = 0 
    correct = 0
    total = 0
    batch_count = 0

    all_test_loss = []  # Stores loss for each item             
    per_image_losses = []  # Collects loss per image for logging

    print(f"Starting evaluation with {len(loader)} batches...")

    # Disable gradient computations for inference
    with torch.no_grad():                          
        for item in tqdm(loader):                  
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"Processing batch {batch_count}/{len(loader)}")

            inputs, labels = item[0], item[1]       
            # Move data to the target device
            inputs = inputs.to(device)
            labels = labels.to(device).to(torch.long)
            
            # Forward pass through the model
            outputs, reps = model(inputs)          
            final_output = outputs[:, 3, :]         # timestep 4

            # Compute batch loss and accumulate 
            loss = criterion(final_output, labels)
            test_loss += loss.item()
            _, predicted = torch.max(final_output.data, 1)

            # Update accuracy metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute the loss (minimizing readout activations)
            #targetloss = target_loss(reps[3])
            #print(targetloss.item())

            # Per-item loss storage 
            for i in range(labels.size(0)):
                item_loss = criterion(final_output[i].unsqueeze(0), labels[i].unsqueeze(0)).item()
                per_image_losses.append({
                    'batch': batch_count,
                    'index_in_batch': i,
                    'true_label': labels[i].item(),   # classe vraie
                    'predicted': predicted[i].item(), # classe prédite
                    'loss': item_loss                 # Loss
                })
                all_test_loss.append(item_loss)

    print(f"Processed {batch_count} batches in total")
    
    # Summary metrics
    avg_test_loss = test_loss / len(loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save results if requested
    if save:
        os.makedirs(save_dir, exist_ok=True)

        # Save summary
        with open(os.path.join(save_dir, 'test_results_gaussian_40.txt'), 'w') as f:
            f.write(f"Test Loss: {avg_test_loss:.4f}\n")
            f.write(f"Accuracy: {accuracy:.2f}%\n")

        # Save per-image loss details
        with open(os.path.join(save_dir, 'per_image_losses_gaussian.csv'), 'w', newline='') as csvfile:
            fieldnames = ['batch', 'index_in_batch', 'true_label', 'predicted', 'loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in per_image_losses:
                writer.writerow(entry)

    return avg_test_loss, accuracy


############################
# ACTIVATION ANALYSIS TOOLS 
############################

# Global storage for cosine similarities
accumulated_results = defaultdict(list)

def analyze_activations(initial_activations, final_activations, class_label=None):
    """
    Compute norms and cosine similarity between two activation vectors.

    Args:
        initial_acts: Tensor of initial activations.
        final_acts: Tensor of final activations.
        class_label: (Optional) Class index for grouping analyses.

    Returns:
        A dict containing initial_norm, final_norm, and cosine_similarity.
    """
    # Convert to numpy if needed
    if torch.is_tensor(initial_activations):
        initial_activations = initial_activations.cpu().detach().numpy()
    if torch.is_tensor(final_activations):
        final_activations = final_activations.cpu().detach().numpy()

    initial_activations = initial_activations.flatten()
    final_activations = final_activations.flatten()
        

    # Calculate norms
    initial_norm = np.linalg.norm(initial_activations)
    final_norm = np.linalg.norm(final_activations)

    # Calculate dot product and cosine similarity
    dot_product = np.dot(initial_activations, final_activations)
    cosine_sim = dot_product / (initial_norm * final_norm + 1e-8)  # small epsilon for stability

    # Store only cosine similarity in the accumulated results
    accumulated_results['cosine_similarities'].append(cosine_sim)

    results = {
        'initial_norm': initial_norm,
        'final_norm': final_norm,
        'cosine_similarity': cosine_sim,
    }
    return results


def save_final_histograms(save_dir, opt_method):
    """
    Génère et sauvegarde un histogramme des similarités cosinus accumulées.

    Args:
        save_dir : Répertoire de base pour les sorties.
        opt_method : Identifiant de la méthode d’optimisation (ex. 'random', 'align').
    """
    
    analysis_dir = os.path.join(save_dir, 'activation_analysis')           
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Cosine Similarity Histogram
    plt.figure(figsize=(10, 6))
    cosines = accumulated_results['cosine_similarities']
    cosines = np.array(cosines).flatten()  # ensures it's 1D
    plt.hist(cosines, bins=10)
    plt.hist(accumulated_results['cosine_similarities'], bins=10)
    plt.title('Distribution of Cosine Similarities Across All Images')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(analysis_dir, f'cosine_similarity_histogram_opt_{opt_method}.png'))  # Comparaison possible : random vs align (pas optimisé pour baseline de comparaison vs masque optimisé pour produire activations ciblées), V1 vs V4, humain vs modèle..
    plt.close()


##########################
# TARGET LOSS CALCULATION 
##########################

def compute_loss(
    outs: torch.Tensor,
    reps: torch.Tensor,
    method: str,
    target_timestep: int = 0,
    init_acts: torch.Tensor = None,
    target_cls: torch.Tensor = None,
    true_cls: torch.Tensor = None,
):
    """
    Select and compute one of several adversarial or analysis losses.

    Methods:
      - 'all_layers': Sum of squared activations across layers.
      - 'target_random_class': Cross-entropy to force a random class.
      - 'align_to_noise_response': MSE between current and noise activations.

    Args:
        outs: Logits tensor of shape [B, T, C] or [B, C].
        reps: List of activation tensors per layer [B, T, ...].
        method: Loss computation strategy.
        target_timestep: Select desired taregt timestep.
        init_acts: Initial activations for matching noise responses.
        target_cls: Randomly sampled class targets.
        true_cls: Ground-truth labels (unused here).
    """

    if method == 'all_layers':
        # compare activations across all layers at target timestep
        if isinstance(reps, list):
            loss = 0.0
            for layer_rep in reps:
                if layer_rep.dim() == 3:
                    act_t = layer_rep[:, target_timestep]
                else:
                    act_t = layer_rep
                loss += torch.mean(act_t**2)  # or MSE wrt zero
            return loss
        else:  # reps simple tensor
            if reps.dim() == 3:
                act_t = reps[:, target_timestep, :]
            else:
                act_t = reps
            return torch.mean(act_t**2)

    elif method == 'target_random_class':
        if outs.dim() == 3:  # [B, T, C]
            logits = outs[:, target_timestep, :]
        elif outs.dim() == 2:
            logits = outs
        else:
            raise ValueError(f"Unexpected logits shape {outs.shape}")

        B = logits.size(0)
        if target_cls is None or target_cls.numel() != B:
            target_cls = torch.randint(0, logits.size(-1), (B,), device=logits.device)
        return F.cross_entropy(logits, target_cls)

    elif method == 'align_to_noise_response':
        if init_acts is None:
            raise ValueError("init_acts must be provided for 'align_to_noise_response'")

        # sécurité : reps peut être 2D ou 3D
        if reps.dim() == 3:  # [B, T, features]
            cur = reps[:, target_timestep, :]
        elif reps.dim() == 2:  # [B, features]
            cur = reps
        else:
            raise ValueError(f"Unexpected reps shape: {reps.shape}")
        
        # === PRINT INTERMEDIAIRE LOSS & COSINE ===
        #print(f"[DEBUG] reps.shape={reps.shape}, cur.shape={cur.shape}, init_acts.shape={init_acts.shape}")

        # MSE entre activations actuelles et activations bruitées
        loss = F.mse_loss(cur, init_acts)
        #print(f"[DEBUG] current loss={loss.item():.6f}")   # valeur scalaire de la loss
        return loss

    else:
        raise ValueError(f"Unrecognized loss method: {method}")

####################
# IMAGE OPTIMIZATION
####################

def optimize_image(
    image: torch.Tensor,
    net: torch.nn.Module,
    target_timestep: int,
    opt_method: str,
    iterations: int,
    batch_idx: int,
    image_idx: int,
    lr: float = 0.01,
    device: str = 'cpu',
    save_dir: str = './results',
    class_label: int = None,
    target_class: int = None,
    save: bool = True
):
    """
    Optimize a mask sequence to achieve a specified internal network response.

    Args:
        image: Input tensor with pre- and post-masks.
        net: Network with `.readout` attribute for classification.
        target_timestep: Desired target timestep to optimize.
        target_layer: Desired target layer to optimize                    # RAJOUTE PAR ADRIEN
        opt_method: Optimization objective name.
        iterations: Number of gradient steps.
        batch_idx: Index for logging filenames.
        image_idx: Index within batch for filenames.
        lr: Learning rate for optimizer.
        device: Compute device ('cpu' or 'cuda').
        save_dir: Root directory to hold output subfolders.
        class_label: If provided, true class for logging.
        save: Whether to persist results to disk.

    Returns:
        seq_final: Final optimized input sequence.
        history: Loss values per iteration.
        init_acts_for_analysis: Activation vector before optimization.
        final_act: Activation vector after optimization.
        analysis_results: Dict with norms and cosine similarity.
    """

    # Prepare output directories
    for d in ['np_masks','loss_hist','timestep_imgs','activation_analysis','metrics']:
        os.makedirs(os.path.join(save_dir, d), exist_ok=True)

    # Separate the image into fixed and optimizable parts
    image = image.to(device).detach()
    premask = image[:, :1]
    mask = image[:, 1].clone().detach().requires_grad_(True)
    postmask = image[:, 3:] if image.shape[1] > 3 else None

    # For random target classification, pre-sample target classes
    sampled_target = None
    if opt_method == 'target_random_class':
        B = image.size(0)
        if target_class is not None:
            sampled_target = torch.full((B,), int(target_class), dtype=torch.long, device=device)
        else: # antoine
            sampled_target = torch.randint(0, net.readout.out_features, (B,), device=device)
        # sampled_target = torch.randint(0, net.readout.out_features, (B,), device=device) # old try

    # Helper to reassemble full sequence from mask
    def assemble(mask_seq, noise=False):
        first = torch.rand_like(premask) if noise else premask
        seq = [first, mask_seq.unsqueeze(1), mask_seq.unsqueeze(1)]
        if postmask is not None:
            seq.append(postmask)
        return torch.cat(seq, dim=1)

    # Compute initial activations for later comparison
    with torch.no_grad():
        seq_init = assemble(mask, noise=False)
        _, reps_init = net(seq_init)
        # Safe selection : if 2D -> no timestep, otherwise takes target_timestep
        if reps_init.dim() == 3:
            init_acts_for_analysis = reps_init[:, target_timestep, :].detach()
        elif reps_init.dim() == 2:
            init_acts_for_analysis = reps_init.detach()
        else:
            raise ValueError(f"Unexpected reps_init shape: {reps_init.shape}")

    # Setup optimizer to adjust only the mask
    optimizer = torch.optim.Adam([mask], lr=lr)

    # If aligning to noise, precompute noise-based activations
    init_acts_for_loss = None
    if opt_method == 'align_to_noise_response':
        with torch.no_grad():
            fine_noise = torch.rand_like(premask) 
            if postmask is not None:
                seq_noise = torch.cat([fine_noise, mask.unsqueeze(1), mask.unsqueeze(1), postmask], dim=1)
            else:
                seq_noise = torch.cat([fine_noise, mask.unsqueeze(1), mask.unsqueeze(1)], dim=1)
            _, reps_noise = net(seq_noise)
            if reps_noise.dim() == 3:
                init_acts_for_loss = reps_noise[:, target_timestep, :].detach()
            elif reps_noise.dim() == 2:
                init_acts_for_loss = reps_noise.detach()
            else:
                raise ValueError(f"Unexpected reps_noise shape: {reps_noise.shape}")

    # Run optimization loop
    history = []
    for i in range(iterations):
        optimizer.zero_grad()
        seq = assemble(mask, noise=False)
        outs, reps = net(seq)

        # Safe selection in compute_loss
        if reps.dim() == 3:
            reps_for_loss = reps
        elif reps.dim() == 2:
            reps_for_loss = reps
        else:
            raise ValueError(f"Unexpected reps shape: {reps.shape}")

        # Compute loss 
        loss = compute_loss(
            outs=outs,
            reps=reps_for_loss,
            method=opt_method,
            init_acts=init_acts_for_loss,
            target_cls=sampled_target,
            true_cls=(torch.tensor(class_label, device=device) if class_label is not None else None),
            target_timestep=target_timestep
        )

        loss.backward()
        optimizer.step()
        mask.data.clamp_(0,1)
        history.append(loss.item())
        if i % 1000 == 0:
            print(f"Batch {batch_idx}, Img {image_idx}, Iter {i}, loss={loss.item():.4f}")

    # Final forward pass 
    seq_final = assemble(mask, noise=False)
    outs_f, reps_f = net(seq_final)
    if reps_f.dim() == 3:
        final_act = reps_f[:, target_timestep, :]
    elif reps_f.dim() == 2:
        final_act = reps_f
    else:
        raise ValueError(f"Unexpected reps_f shape: {reps_f.shape}")

    # Analyze activation changes
    analysis_results = analyze_activations(init_acts_for_analysis, final_act)


    ## Save metrics and results if requested
    if save:
        # Save metrics file
        metrics_path = os.path.join(save_dir, 'metrics', f'all_metrics_{opt_method}.txt')  # METRICS FOLDER
        with open(metrics_path, 'a') as f:
            f.write(f"Batch {batch_idx}, Image {image_idx} ({opt_method}):")
            # Cosine similarity if available
            cs = analysis_results.get('cosine_similarity', None)
            if cs is not None:
                try:
                    mean_cs = float(cs) if not hasattr(cs, 'mean') else float(cs.mean())
                    f.write(f"  Cosine Similarity: {mean_cs:.6f}")
                except:
                    pass
            # Final loss
            final_loss = history[-1] if history else float('nan')
            f.write(f"  Final Loss: {final_loss:.6f}")
            f.write("-"*40 + "")

        # Save loss history
        prefix = f"b{batch_idx}_i{image_idx}_{opt_method}"
        hist_arr = torch.tensor(history).cpu().numpy()
        loss_hist_dir = os.path.join(save_dir, 'loss_hist')                               # LOSS_HIST FOLDER
        torch.save(hist_arr, os.path.join(loss_hist_dir, f'hist_{prefix}.pt'))
        with open(os.path.join(loss_hist_dir, f'hist_{prefix}.txt'), 'w') as fh:
            for v in history:
                fh.write(f"{v}\n")

        # Save optimized mask as npy array                                                # NP_MASKS FOLDER
        np_masks_dir = os.path.join(save_dir, 'np_masks')
        mask_final = mask.unsqueeze(1)
        mask_img = mask_final[0,0].permute(1,2,0).cpu().detach().numpy() if mask_final.dim()==4 else mask_final[0].cpu().detach().numpy()
        np.save(os.path.join(np_masks_dir, f'mask_b{batch_idx}_i{image_idx}_{opt_method}.npy'), mask_img)  # mask_bX_iX_opt_method.npy

        # Save optimized mask as image PNG                                                # PLT_MASKS FOLDER
        plt_masks_dir = os.path.join(save_dir, 'plt_masks')
        os.makedirs(plt_masks_dir, exist_ok=True)

        # Prépare le masque
        mask_final = mask.unsqueeze(1) if mask.dim() == 3 else mask  # force dimension (B,C,H,W) si nécessaire          
        mask_img = mask_final[0].cpu().detach().numpy()  # (C,H,W) ou (H,W)
        if mask_img.ndim == 3 and mask_img.shape[0] in [1,3]:
             mask_plt = np.transpose(mask_img, (1,2,0))  # (H,W,C) pour matplotlib
        elif mask_img.ndim == 2:
            mask_plt = mask_img
        else:
            raise ValueError(f"Shape du masque inattendue : {mask_img.shape}")

        existing_files = [f for f in os.listdir(plt_masks_dir) if f.startswith(f"mask_b{batch_idx}_i{image_idx}_opt-{opt_method}")]  # Numérotation automatique pour éviter écrasement
        file_idx = len(existing_files) + 1

        filename = f"mask_b{batch_idx}_i{image_idx}_opt-{opt_method}_{file_idx}.png"
        file_path = os.path.join(plt_masks_dir, filename)

        # Conserve la taille originale du masque
        H, W = mask_plt.shape[:2]
        fig = plt.figure(figsize=(W/100, H/100), dpi=100) # figsize*dpi = pixels
        plt.imshow(mask_plt)
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # supprime marges
        fig.savefig(file_path, dpi=100)
        plt.close()

        # Save only the target timestep image                                             # TIMESTEP_IMG FOLDER
        timestep_dir = os.path.join(save_dir, 'timestep_imgs')
        img_t = seq_final[0, target_timestep].permute(1, 2, 0).cpu().detach().numpy()
        np.save(os.path.join(timestep_dir, f'ts{target_timestep}_b{batch_idx}_i{image_idx}_{opt_method}.npy'), img_t)

    # Return final results
    return seq_final, history, init_acts_for_analysis, final_act, analysis_results



#################################
# DATASET SAMPLING BY LUMINANCE
#################################

def sample_images_by_luminance(
    dataset_path,                 
    classes_to_use,                
    n_per_class=50,                
    seed=0,                                           
    output_root='selected_images',         
    metrics_file='selected_metrics.csv',    
    plots_dir='plots',                    
    dataset_split="test",         
    luminance_filter_sigma=1.0
):
    """
    Select and save images filtered by mean luminance within a dataset.

    Steps:
      1. Load dataset and compute per-image luminance.
      2. Compute global luminance mean & std dev.
      3. Filter images within ±sigma range for each class.
      4. Randomly sample up to n_per_class per class.
      5. Save images to class-labeled folders.
      6. Write sampling metrics to CSV.
      7. Plot and save overall luminance histogram.

    Returns:
        Dictionary with sampled data and summary statistics.
    """
    
    # 1) Load raw dataset   
    raw_ds = H5Dataset(
        dataset_path,
        split=dataset_split,
        transform=transforms.ToTensor()
    )
    
    # 2) Compute per-image luminance
    lum_w = torch.tensor([0.2126, 0.7152, 0.0722]).view(3, 1, 1)
    records = []
    print("Computing mean luminance for each target image...")
    
    for idx in range(len(raw_ds)):
        img, label = raw_ds[idx]
        if label not in classes_to_use:
            continue
        Y = (img * lum_w).sum(dim=0).mean().item()                
        records.append({'idx': idx, 'class_id': label, 'lum': Y}) 
    
    # 3) Global mean & std dev
    lums = np.array([r['lum'] for r in records])
    mu, sd = lums.mean(), lums.std()
    low, high = mu - (luminance_filter_sigma * sd), mu + (luminance_filter_sigma * sd) 
    print(f"Luminance: mean={mu:.3f}, std={sd:.3f}, keeping in [{low:.3f},{high:.3f}]")
    
    # 4) Filter & sample per class                      
    print("Filtering and sampling images per class...")
    rng = random.Random(seed)
    sampled = {}
    
    for c in classes_to_use:
        filtered = [r for r in records if r['class_id'] == c and low <= r['lum'] <= high]
        idxs = [r['idx'] for r in filtered]
        
        if len(idxs) < n_per_class:
            print(f"Warning: only {len(idxs)} images pass filter in class {c}")
        
        chosen_idxs = rng.sample(idxs, min(n_per_class, len(idxs)))
        sampled[c] = [r for r in filtered if r['idx'] in chosen_idxs]
        print(f"Class {c}: selected {len(sampled[c])} images")
    
    # 5) Save sampled images into per-class folders  
    os.makedirs(output_root, exist_ok=True)
    
    for c, recs in sampled.items():
        class_dir = os.path.join(output_root, f'class_{c}')
        os.makedirs(class_dir, exist_ok=True)
        
        for r in recs:
            idx = r['idx']
            img, *_ = raw_ds[idx]  # Fixed the syntax error here
            img_pil = transforms.ToPILImage()(img)
            img_pil.save(os.path.join(class_dir, f'img_{idx}.png'))
    
    print(f"Saved sampled images into subfolders of {output_root}/")
    
    # 6) Save metrics to CSV                                  
    with open(metrics_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['class_id', 'idx', 'lum'])
        writer.writeheader()
        for c, recs in sampled.items():
            for r in recs:
                writer.writerow({'class_id': r['class_id'], 'idx': r['idx'], 'lum': r['lum']})
    
    print(f"Saved selected metrics to {metrics_file}")
    
    # 7) Plot & save luminance histogram
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure()
    plt.hist(lums, bins=50)
    plt.axvline(mu, color='k', linestyle='-', label=f"μ={mu:.2f}")
    plt.axvline(low, color='k', linestyle='--', label=f"μ–σ={low:.2f}")
    plt.axvline(high, color='k', linestyle='--', label=f"μ+σ={high:.2f}")
    plt.legend()
    plt.title("Luminance Distribution (Mean ± 1σ)")
    plt.xlabel("Mean Luminance")
    plt.ylabel("Count")
    
    histogram_path = os.path.join(plots_dir, 'luminance_histogram.png')
    plt.savefig(histogram_path)
    plt.close()
    print(f"Saved luminance histogram to {histogram_path}")
    
    # Prepare return data
    total_sampled = sum(len(recs) for recs in sampled.values())
    
    return {
        'sampled_data': sampled,
        'luminance_stats': {
            'mean': mu,
            'std': sd,
            'low_threshold': low,
            'high_threshold': high
        },
        'total_images_sampled': total_sampled
    }