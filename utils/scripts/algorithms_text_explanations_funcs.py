import json
from dataclasses import dataclass
from typing import List
import pandas as pd
from tabulate import tabulate
import matplotlib.lines as mlines
import torch
import numpy as np
from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, ImageFolder
from utils.datasets.binary_waterbirds import BinaryWaterbirds

import random
from collections import defaultdict
from utils.datasets.dataset_helpers import dataset_subset
import matplotlib.pyplot as plt
from utils.misc.visualization import visualization_preprocess
from utils.misc.misc import accuracy_correct
from utils.datasets_constants.imagenet_classes import imagenet_classes
from utils.datasets_constants.cifar_10_classes import cifar_10_classes
from utils.datasets_constants.waterbird_classes import cub_classes, waterbird_classes


### Layout of data
@dataclass
class PrincipalComponentRecord:
    layer: int
    head: int
    princ_comp: int
    strength_abs: float
    strength_rel: float
    cosine_similarity: float
    texts: List[str]
    project_matrix: torch.Tensor
    vh: torch.Tensor
    rank: int

# Fix seeds
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
@torch.no_grad()
def get_data(attention_dataset, min_princ_comp=-1, skip_final=False):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - attention_dataset (str): The path to the JSON file containing attention data.
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).

    Returns:
    - A list of dictionaries containing details for each principal component of interest.
    """
    with open(attention_dataset, "r") as json_file:
        data = []
        for line in json_file:
            entry = json.loads(line)  # Parse each line as a JSON object, producing a dictionary-like structure
             # Skip the final clip embeddings if requested
            if skip_final and entry["head"] == -1:
                continue

            # Each entry includes a sorted list of principal components. 
            # We want to record information up to a certain minimum principal component index (min_princ_comp).
            for i, princ_comp_data in enumerate(entry["embeddings_sort"]):
                # Stop if we've reached the minimum principal component index to limit how many we gather from each entry
                if i == min_princ_comp:
                    break
               
                # Append a dictionary of details for each principal component of interest
                data.append({
                    "layer": entry["layer"],
                    "head": entry["head"],
                    "princ_comp": i,
                    "strength_abs": princ_comp_data["strength_abs"],
                    "strength_rel": princ_comp_data["strength_rel"],
                    "cosine_similarity": princ_comp_data["cosine_similarity"],
                    "correlation": princ_comp_data["correlation"],
                    "texts": princ_comp_data["text"],
                    "rank": len(entry["embeddings_sort"]),
                    "project_matrix": entry["project_matrix"],
                    "vh": entry["vh"],
                    "mean_values_text": entry["mean_values_text"],
                    "mean_values_att": entry["mean_values_att"],
                })
        
    return data

@torch.no_grad()
def get_data_component(data, layer, head, princ_comp):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of dictionaries containing detail for the principal component of interest.
    """

    for entry in data:
        if entry["layer"] == layer and entry["head"] == head and entry["princ_comp"] == princ_comp:
            return [entry]


def plot_pc_sv_across_heads_and_layers(data, base_layer, nr_layer=4):
    """
    Plot the cumulative singular values (using cumulative sum of 'strength_rel') for multiple layers 
    (the base layer and the three previous layers). For each layer, we aggregate over all heads in the 
    data, compute the mean cumulative singular value at each principal component index, and shade the region 
    within 1 standard deviation above and below the mean. The legend for each layer includes the percentage 
    of variance explained by the first PC. The shadow areas are made more distinct by increasing the 
    opacity and drawing boundary lines.

    Args:
        data (list): A list of dictionaries containing the principal component records.
        base_layer (int): The highest layer (the “value layer”) from which to start plotting. 
                          The plot will include base_layer, base_layer-1, base_layer-2, and base_layer-3.

    Returns:
        None
    """
    layers = [base_layer - x for x in range(nr_layer)]  #rs)

    plt.figure(figsize=(10, 6))
    
    # Loop through each layer and aggregate across heads.
    for l in layers:
        # Determine which heads exist for the given layer
        heads = sorted(set(entry["head"] for entry in data if entry["layer"] == l))
        
        # Lists to hold the cumulative singular value arrays and PC0 percentages for each head
        cum_svs_list = []
        pc0_vals = []  # This will store the first singular value for each head
        
        for head in heads:
            svs = get_data_head_svs(data, l, head)
            if not svs:
                continue
            
            # Collect the first singular value (assumed to represent PC0's variance explained)
            pc0_vals.append(svs[0])
            # Compute cumulative singular values
            cum_svs = np.cumsum(svs)
            cum_svs_list.append(cum_svs)
        
        # Skip the layer if no valid heads are found.
        if len(cum_svs_list) == 0:
            continue

        # Ensure all arrays have the same length (truncate to the smallest length)
        min_length = min(len(arr) for arr in cum_svs_list)
        cum_svs_list = [arr[:min_length] for arr in cum_svs_list]
        cum_svs_array = np.array(cum_svs_list)  # shape: (num_heads, num_components)
        
        # Compute the mean and standard deviation across heads at each principal component index.
        mean_sv = np.mean(cum_svs_array, axis=0)
        std_sv = np.std(cum_svs_array, axis=0)

        # X axis will be the principal component index.
        x = np.arange(1, min_length + 1)
        
        # Calculate the average percentage of variance explained by the first PC (PC0)
        avg_pc0 = np.mean(pc0_vals)
        # Format the legend label as "Layer {l} (PC0: 9%)" rounded to no decimal places.
        label_str = f'Layer {l} (PC0: {avg_pc0:.0f}%)'
        
        # Plot the mean cumulative singular values with the custom legend label.
        line, = plt.plot(x, mean_sv, label=label_str)
        
        # Increase the opacity of the filled region (e.g., alpha=0.4)
        plt.fill_between(x, mean_sv - std_sv, mean_sv + std_sv, color=line.get_color(), alpha=0.1)
        
        # Add boundary lines to the fill for clarity
        plt.plot(x, mean_sv - std_sv, linestyle='--', color=line.get_color(), linewidth=1, alpha=0.15)
        plt.plot(x, mean_sv + std_sv, linestyle='--', color=line.get_color(), linewidth=1, alpha=0.15)
        
    plt.xlabel('Principal Component Index')
    plt.ylabel('Cumulative Variance')
    plt.title('Mean Cumulative Variance (PCs) Across Heads of Last Four Layers with ±1 Std')
    plt.legend(title="Layer")
    plt.grid(True)
    plt.show()

    
def print_data_text_span(data, top_k=None):
    """
    Print the collected data in a formatted table.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    Returns:
    - None
    """
    # Convert the collected data into a Pandas DataFrame for easier manipulation and printing
    top_k_df = pd.DataFrame(data)

    # Iterate over each row in the DataFrame to display details about the principal components and their associated texts
    for row in top_k_df.itertuples():
        output_rows = []
        texts = row.texts
        
        if top_k != None:
            output_rows = texts[:top_k]
        else:
            output_rows = texts

        # Print summary information about the current principal component:
        # Including layer, head, principal component index, absolute variance, relative variance, and head rank.
        print(f"Layer {row.layer}, Head {row.head}")
        # Set the column headers based on whether the first half was considered positive
        columns = ["text"]

        # Create a DataFrame from the collected rows of positive/negative texts and print it in a formatted table
        output_df = pd.DataFrame(output_rows, columns=columns)
        print(tabulate(output_df, headers='keys', tablefmt='psql'))

def visualize_text_span(layer, head, data, top_k=-1):    # Retrieve and filter the principal component data
    data = get_data_text_span(data)
    data = get_data_head(data, layer, head)
    print_data_text_span(data, top_k=top_k)


def get_data_text_span(attention_dataset):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - attention_dataset (str): The path to the JSON file containing attention data.
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).

    Returns:
    - A list of dictionaries containing details for each principal component of interest.
    """
    with open(attention_dataset, "r") as json_file:
        data = []
        for line in json_file:
            entry = json.loads(line)  # Parse each line as a JSON object, producing a dictionary-like structure


            # Each entry includes a sorted list of principal components. 
            # We want to record information up to a certain minimum principal component index (min_princ_comp).

               
            # Append a dictionary of details for each principal component of interest
            data.append({
                "layer": entry["layer"],
                "head": entry["head"],
                "texts": entry["embeddings_sort"]
            })
    
    return data

def get_data_head_svs(data, layer, head):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of svs
    """
    svs = []
    for entry in data:
        if entry["layer"] == layer and entry["head"] == head:
           svs.append(entry["strength_rel"]) 
    return svs

def plot_pc_sv(data, layer, head):
    """
    Plot the singular values of the principal components for a given layer and head.

    Args:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.

    Returns:
    - None
    """
    # Retrieve and filter the principal component data
    svs = get_data_head_svs(data, layer, head)
    svs = np.cumsum(svs)
    # Plot the singular values of the principal components on the y axis
    plt.figure(figsize=(10, 6))
    plt.plot(svs, marker='o')
    plt.xlabel('Principal Component Index')
    plt.ylabel('Cumulative Singular Value')
    plt.title(f'Cumulative Singular Values for Layer {layer}, Head {head}')
    plt.grid(True)
    plt.show()



def get_data_head(data, layer, head):
    """
    Retrieve data from a JSON file containing attention data.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - layer (int): The layer number of the principal component.
    - head (int): The head number of the principal component.
    - princ_comp (int): The principal component number of interest.

    Returns:
    - A list of dictionaries containing detail for the principal component of interest.
    """
    for entry in data:
        if entry["layer"] == layer and entry["head"] == head:
            return [entry]
    return []


@torch.no_grad()
def print_data(data, min_princ_comp=-1, is_corr_present=False):
    """
    Print the collected data in a formatted table.
    Args:
    - data (list): A list of dictionaries containing details for each principal component of interest. (i.e. layout as PrincipalComponentRecord)
    - min_princ_comp (int): The minimum principal component number to consider for each head (-1=all).
    - is_corr_present (bool): Whether the correlation values are present in the data.
    Returns:
    - None
    """
    # Convert the collected data into a Pandas DataFrame for easier manipulation and printing
    top_k_df = pd.DataFrame(data)

    # Iterate over each row in the DataFrame to display details about the principal components and their associated texts
    for row in top_k_df.itertuples():
        output_rows = []
        texts = row.texts
        half_length = len(texts) // 2
        
        # Skip the enetry if above the minimum number we want
        if row.princ_comp >= min_princ_comp and min_princ_comp != -1:
            continue

        # Determine if the first half of the texts represent a positive component by checking the associated value
        is_positive_first = list(texts[0].values())[1] > 0
        
        # Split the texts into two halves: first half (positive or negative) and second half (the opposite polarity)
        if is_positive_first:
            positive_texts = texts[:half_length]
            negative_texts = texts[half_length:]
        else:
            positive_texts = texts[half_length:]
            negative_texts = texts[:half_length]
        
        # Pair up corresponding positive and negative texts and collect them in output_rows for tabular display
        for pos, neg in zip(positive_texts, negative_texts):
            pos_text = list(pos.values())[0]
            pos_val  = list(pos.values())[1]
            neg_text = list(neg.values())[0]
            neg_val  = list(neg.values())[1]    
            
            # Append not with maximum value but with the according sign of correlation on the left
            if is_corr_present:
                corr_sign = np.sign(row.correlation_princ_comp)
                if corr_sign > 0:
                    output_rows.append([pos_text, pos_val, neg_text, neg_val])
                else:
                    output_rows.append([neg_text, neg_val, pos_text, pos_val])

            else:
                if is_positive_first:
                    output_rows.append([pos_text, pos_val, neg_text, neg_val])
                else:
                    output_rows.append([neg_text, neg_val, pos_text, pos_val])            

        # Print summary information about the current principal component:
        # Including layer, head, principal component index, absolute variance, relative variance, and head rank.
        print(f"Layer {row.layer}, Head {row.head}, Principal Component {row.princ_comp}, "
            f"Variance {row.strength_abs:.3f}, Relative Variance {row.strength_rel:.3f}, Head Rank {row.rank}")
        if is_corr_present:
            print(f"Correlation of the Topic with the Principal Component {row.correlation_princ_comp:.4f}")
        
        # Set the column headers based on whether the first half was considered positive
        if is_positive_first:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]

        # Or set it based on the sign of the correlation
        if is_corr_present and corr_sign > 0:
            columns = ["Positive", "Positive_Strength", "Negative", "Negative_Strength"]
        else:
            columns = ["Negative", "Negative_Strength", "Positive", "Positive_Strength"]
        
        # Create a DataFrame from the collected rows of positive/negative texts and print it in a formatted table
        output_df = pd.DataFrame(output_rows, columns=columns)
        print(tabulate(output_df, headers='keys', tablefmt='psql'))

@torch.no_grad()
def sort_data_by(data, key="strength_abs", descending=True):
    """
    Sorts data based on the 'strength_abs' key.

    Parameters:
    - data (list): List of dictionaries to sort.
    - key (str): Key to sort by.
    - descending (bool): Sort in descending order if True, ascending otherwise.

    Returns:
    - list: Sorted list of data.
    """
    return sorted(data, key=lambda x: x.get(key, 0), reverse=descending)

@torch.no_grad()
def top_data(data, top_k=5):
    """
    Get the top-k elements of data (already sorted)

    Parameters:
    - data (list): List of dictionaries from which to retrieve top elements.
    - top_k (int): Number of top-k elements to retrieve.

    Returns:
    - list: Top-k data.
    """
    return data[:top_k]

@torch.no_grad()
def map_data(data, lbd_func=None):
    """
    Apply lambda function on each element of data.

    Parameters:
    - data (list): List of dictionaries from which to retrieve top elements.
    - lbd_func (x: x): Lambda function.

    Returns:
    - list: Mapped elements.
    """
    return [lbd_func(x) for x in data]

@torch.no_grad()
def reconstruct_all_embeddings_mean_ablation_pcs(data_pcs, mlps, attns, attns_dataset, tot_nr_layers, tot_nr_heads, nr_mean_ablated, ratio=-1, mean_ablate_all=False, ablation=True, return_attention=False):
    """
    Reconstruct the embeddings using the contribution of only the heads with PCs in data.
    Parameters:
    - data (list): List of dictionaries containing all details for each principal component.
    - data_pcs (list): List of dictionaries containing all details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings. mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))


    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    embeddings = mlps.sum(axis=1) + attns.sum(axis=(1, 2))

    # Clone attns
    attns = attns.clone()
    # Sort content of data_pcs based on the natural order of pcs
    heads_to_keep = sorted(
    list(
        set((x["layer"], x["head"]) for x in data_pcs if (x["layer"], x["head"]) != (-1, -1))
        )
    )
    print("Heads to keep: ", len(heads_to_keep))
    
    grouped_data = []

    for (layer, head) in heads_to_keep:
        # Filter the original data to only those items matching the current layer/head
        relevant_items = [item for item in data_pcs if item["layer"] == layer and item["head"] == head]
        # Collect all principal components, vh, and mean_values for the current layer/head
        princ_comps = [item["princ_comp"] for item in relevant_items]
        s = torch.tensor([item["strength_abs"] for item in relevant_items])
        vh_vals     = torch.tensor(relevant_items[0]["vh"])
        means       = torch.tensor(relevant_items[0]["mean_values_att"]).unsqueeze(0)

        grouped_data.append(
            (layer,
            head,
            princ_comps,
            s,
            vh_vals,
            means)
            )


    # Initialize the reconstructed embeddings
    nr_layer_not_anal = tot_nr_layers - nr_mean_ablated
    # Put initial contribution
    prev_layers = mlps.sum(axis=1) + attns[:, 0:nr_layer_not_anal, ...].sum(axis=(1, 2))
    reconstructed_embeddings = torch.zeros_like(prev_layers)
    reconstruct_embeddings_mean_ablate = torch.zeros_like(prev_layers)
    # Iterate over each principal component of interest and keep the contribution of their head, sum mean ablation for all the other
    prev_layer, prev_head = nr_layer_not_anal, 0
    for layer, head, princ_comps, s, vh, mean_values in grouped_data:
        # Get contribution for our principal component on mean centered data
        mask = torch.zeros((reconstructed_embeddings.shape[0], vh.shape[0]))
        mask_mean = torch.zeros((1, vh.shape[0]))
        if return_attention:
            mask = torch.zeros((reconstructed_embeddings.shape[0], attns.shape[3], vh.shape[0]))
            mask_mean = torch.zeros((1, attns.shape[3], vh.shape[0]))
            nr_patches = attns.shape[3]

        if not ablation:
            mask = ((attns[:, prev_layer, prev_head, ...]) @ vh.T)
            reconstructed_embeddings += mask @ vh

            mask_mean = ((- mean_values) @ vh.T)
            reconstruct_embeddings_mean_ablate += mask_mean @ vh

        else:
            mask[..., princ_comps] = ((attns[:, prev_layer, prev_head, ...]) @ vh.T)[..., princ_comps]
            reconstructed_embeddings += mask @ vh
    
            mask_mean[..., princ_comps] = ((- torch.mean(attns_dataset[:, prev_layer, prev_head, ...], dim=0).unsqueeze(0)) @ vh.T)[..., princ_comps]
            reconstruct_embeddings_mean_ablate += mask_mean @ vh

        if return_attention:
            tmp= mask @ vh + (mask_mean @ vh)/nr_patches + torch.mean(attns_dataset[:, prev_layer, prev_head, ...], dim=0)/nr_patches
            attns[:, layer, head, :] = tmp


        reconstruct_embeddings_mean_ablate += torch.mean(attns_dataset[:, prev_layer, prev_head, ...], dim=0)

        # Add mean ablation for whole heads when not used 
        while prev_layer != layer or prev_head != head: 
            # Add mean contribution of all the data not in the pcs
            if ablation:
                reconstruct_embeddings_mean_ablate += torch.mean(attns_dataset[:, prev_layer, prev_head, ...], axis=0) #[b, l, h, d]
                if return_attention:
                    attns[:, prev_layer, prev_head, :] = 0#torch.mean(attns_dataset[:, prev_layer, prev_head, ...], axis=0)/nr_patches
            else:
                reconstructed_embeddings += attns[:, prev_layer, prev_head, ...]

            if prev_head == tot_nr_heads - 1:
                prev_head = 0
                prev_layer += 1
            else:
                prev_head += 1   
            if prev_head == tot_nr_heads -1 and prev_layer == tot_nr_layers - 1:
                break
        prev_head, prev_layer = head + 1, layer
        if prev_head == tot_nr_heads:
                prev_head = 0
                prev_layer += 1
        if prev_head == tot_nr_heads and prev_layer == tot_nr_layers:
            break
    
    # Amplification ==-1 keep original ratio
    if ratio == -1:
        # If we want to mean ablate all previous values
        if mean_ablate_all:
            reconstructed_embeddings += torch.mean(prev_layers + reconstruct_embeddings_mean_ablate, dim=0)
        else:
            reconstructed_embeddings += prev_layers + reconstruct_embeddings_mean_ablate
    else:
        # Consider as a whole prev layers and mean ablation
        reconstruct_embeddings_mean_ablate += prev_layers
        norm_rec = reconstructed_embeddings.norm(dim=-1, keepdim=True)
        norm_rec_mean = reconstruct_embeddings_mean_ablate.norm(dim=-1, keepdim=True)
        norm_mean = torch.mean(embeddings, axis = 0).norm(dim = -1).item()
        reconstructed_embeddings *= norm_mean / norm_rec * ratio
        reconstructed_embeddings += reconstruct_embeddings_mean_ablate*norm_mean / norm_rec_mean * (1-ratio)
        
    # No pcs selected, just return whole mean ablation
    if data_pcs == []:
        reconstructed_embeddings = mlps.sum(axis=1) + torch.mean(attns_dataset.sum(axis=(2))[:, :], axis=0).sum(0)
    
    if return_attention:
        return attns
    else:
        return reconstructed_embeddings

@torch.no_grad()
def reconstruct_all_embeddings_mean_ablation_heads(data_pcs, mlps, attns, embeddings, tot_nr_layers, tot_nr_heads, nr_mean_ablated):
    """
    Reconstruct the embeddings using the contribution of only the heads with PCs in data.
    Parameters:
    - data (list): List of dictionaries containing all details for each principal component.
    - data_pcs (list): List of dictionaries containing all details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings. mean_ablated_and_replaced = mlps.sum(axis=1) + attns.sum(axis=(1, 2))


    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0:
        assert False, "No embeddings to reconstruct or different lengths."

    # Sort content of data_pcs based on the natural order of pcs
    heads_to_keep = sorted(list(set([(x["layer"], x["head"]) for x in data_pcs if (x["layer"], x["head"]) != (-1, -1)])))

    print("Heads to keep:", len(heads_to_keep))
    # Initialize the reconstructed embeddings
    nr_layer_not_anal = tot_nr_layers - nr_mean_ablated
    # Put initial contribution
    reconstructed_embeddings = mlps.sum(axis=1) + attns[:, 0:nr_layer_not_anal, :, :].sum(axis=(1, 2))

    # Iterate over each principal component of interest and keep the contribution of their head, sum mean ablation for all the other
    prev_layer, prev_head = nr_layer_not_anal, 0
    for layer, head in heads_to_keep:
        reconstructed_embeddings += attns[:, layer, head, :]
        # Add mean ablation
        while prev_layer != layer or prev_head != head: 
            # Add mean contribution of all the data not in the pcs
            reconstructed_embeddings += torch.mean(attns[:, prev_layer, prev_head, :], axis=0) #[b, l, h, d]
            if prev_head == tot_nr_heads - 1:
                prev_head = 0
                prev_layer += 1
            else:
                prev_head += 1   
            if prev_head == tot_nr_heads -1 and prev_layer == tot_nr_layers - 1:
                break
        prev_head, prev_layer = head + 1, layer
        if prev_head == tot_nr_heads:
                prev_head = 0
                prev_layer += 1
        if prev_head == tot_nr_heads and prev_layer == tot_nr_layers:
            break

    return reconstructed_embeddings

def random_pcs(data, nr_pcs):
    """
    Randomly select principal components from the data.
    """
    random_pcs = np.random.choice(data, nr_pcs, replace=False)
    return list(random_pcs.squeeze())


def select_random_pcs(data, n, layers="late", seed=0, late_k=4):
    """
    Random-PC control for the QuerySystem (paper section 6.4(ii)).

    Returns a selection with the same shape as the QuerySystem selection
    (a list of `n` PrincipalComponentRecord-style dicts) so that the existing
    reconstruction path (`reconstruct_embeddings`) can be reused unchanged.

    Parameters:
    - data (list): PC records as produced by `get_data`.
    - n (int): number of PCs to select.
    - layers ('late' | 'all'): restrict the candidate pool to the last `late_k`
      layers ('late') or draw uniformly across all layers present in `data` ('all').
      Note the explanation .jsonl already only contains the last
      `--num_of_last_layers`; 'late' narrows this further.
    - seed (int): RNG seed (independent of the global seed).
    - late_k (int): number of trailing layers considered 'late'.

    Returns:
    - list of `n` PC records drawn without replacement.
    """
    if layers == "late":
        max_layer = max(entry["layer"] for entry in data)
        candidates = [e for e in data if e["layer"] >= max_layer - late_k + 1]
    elif layers == "all":
        candidates = list(data)
    else:
        raise ValueError(f"layers must be 'late' or 'all', got {layers!r}")

    if n > len(candidates):
        raise ValueError(f"Requested {n} PCs but only {len(candidates)} candidates for layers={layers!r}.")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in idx]

@torch.no_grad()
def reconstruct_embeddings(data, embeddings, types, device="cpu", return_princ_comp=False, plot=False, means=[]):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings.

    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0 or len(types) != len(embeddings):
        assert False, "No embeddings to reconstruct or different lengths."

    # Initialize the reconstructed embeddings
    reconstructed_embeddings = [torch.zeros_like(embeddings[i].to(device)) for i in range(len(embeddings))]

    # Store reconstruction values for plotting
    rec_x = []
    num_elements = []

    # Iterate over each principal component of interest
    for count, component in enumerate(data):
        # Retrieve projection matrices and mean values
        project_matrix = torch.tensor(component["project_matrix"]).to(device)
        vh = torch.tensor(component["vh"]).to(device)
        princ_comp = component["princ_comp"]
        s = component["strength_abs"]

        # Reconstruct Embeddings
        for i in range(len(embeddings)):
            # Derive masking
            mask = torch.zeros((embeddings[i].shape[0], vh.shape[0])).to(device)

            if types[i] == "text":
                mask[:, princ_comp] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze()
                embed_proj_back = mask @ project_matrix @ vh
                reconstructed_embeddings[i] += embed_proj_back
            else:
                mask[:, princ_comp] = (embeddings[i] @ vh.T)[:, princ_comp].squeeze()
                embed_proj_back = mask @ project_matrix @ vh
                reconstructed_embeddings[i] += embed_proj_back

            if plot:
                reconstructed_embeddings_norm = reconstructed_embeddings[i] / reconstructed_embeddings[i].norm(dim=-1, keepdim=True)
                embedding_norm = embeddings[i] / embeddings[i].norm(dim=-1, keepdim=True)
                # Calculate and store the cosine reconstruction score
                cosine_score = reconstructed_embeddings_norm @ (embedding_norm).T
                rec_x.append(cosine_score.item())
                num_elements.append(count)

            if return_princ_comp:
                component["correlation_princ_comp_abs"] = torch.abs(mask[:, princ_comp]).item()
                component["correlation_princ_comp"] = mask[:, princ_comp].item()

    # Plot the reconstruction values if requested
    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(num_elements, rec_x, marker='o')
        plt.ylabel("Reconstructed Cosine Similarity")
        plt.xlabel("Number of Principal Components Used")
        plt.title("Reconstruction Cosine Similarity vs. Number of Principal Components")
        plt.grid(True)
        plt.show()

    return reconstructed_embeddings, data

def print_used_heads(data):
    """
    Print the used heads in the data.

    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.

    Returns:
    - None
    """
    # Track Found heads and layers
    track_h_l = {}
    for entry in data:
        key = (entry["layer"], entry["head"])
        val = track_h_l.get(key, 0)
        track_h_l[key] = val
        track_h_l[key] += 1

    print(f"Used a total number of {len(track_h_l)} different heads")


    return data

def get_remaining_pcs(data, top_k_details):
    # Get the remaining principal components (very simple and unoptimized)
    remaining_pcs = []
    for entry_data in data:
        found = False
        for entry in top_k_details:
            layer = entry["layer"]
            head = entry["head"]
            pc = entry["princ_comp"]

            if entry_data["layer"] == layer and entry_data["head"] == head and entry_data["princ_comp"] == pc:
                found = True
                break
        if not found:
            remaining_pcs.append(entry_data)

    return remaining_pcs


def reconstruct_top_embedding(data, embedding, mean, type, max_reconstr_score, top_k=10, approx=0.90, plot=True, device="cpu"):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
        - data (list): List of dictionaries containing details for each principal component of interest.
        - embedding: The embedding to reconstruct.
        - type: Type of embedding to reconstruct.
        - max_reconstr_score: Maximum achievable reconstruction score.
        - top_k: Number of top principal components to consider.
        - approx: Approximation threshold for the reconstruction score.

    Returns:
        - data: Data updated with top_k principal components of the given embeddings (if return_princ_comp is True).
    """

    # Store reconstruction values for plotting
    rec_x = []
    num_elements = []
    # Track Found heads and layers
    track_h_l = {}
    
    for count, entry in enumerate(data):
        if count == 0:
            # For the first principal component, initialize the query representation
            [query_repres], _ = reconstruct_embeddings([data[count]], [embedding], [type], return_princ_comp=False, device=device)

        else:
            # For subsequent components, accumulate their contributions
            [query_repres_tmp], _ = reconstruct_embeddings([data[count]], [embedding], [type], return_princ_comp=False, device=device)
            query_repres += query_repres_tmp
        # Add count of layer and head
        key = (data[count]["layer"], data[count]["head"])
        val = track_h_l.get(key, 0)
        track_h_l[key] = val
        track_h_l[key] += 1

        # Compute the current score:
        query_repres_norm = query_repres/query_repres.norm(dim=-1, keepdim=True)
        
        embedding_norm = embedding/embedding.norm(dim=-1, keepdim=True)
        # Compute the current score: how well this partial reconstruction matches the original embedding
        score = query_repres_norm @ embedding_norm.T
        #score = query_repres_norm @ (embedding + mean).T

        
        rec_x.append(score.item())
        num_elements.append(count)

        # If we've reached the top_k limit or our reconstruction score is good enough 
        # (relative to max_reconstr_score and meeting the approx threshold), stop adding more components.
        if count == (top_k - 1) or score / max_reconstr_score > approx:
            top_k = count + 1
            break

    # Print information about how well the reconstruction performed
    percentage = (score / max_reconstr_score * 100).item()
    print(
        f"Reconstruction Quality Report:\n"
        f"- Maximum achievable reconstruction score: {max_reconstr_score.item():.4f}\n"
        f"- Used a total number of {len(track_h_l)} different heads\n"
        f"- Current reconstruction score: {score.item():.4f}\n"
        f"  This corresponds to {percentage:.2f}% of the maximum possible score.\n"
    )

    print(
        f"The reconstruction was performed using the top {top_k} principal component(s).\n "
        f"Increasing this number may improve the reconstruction score.\n\n"
    )

    if plot:
        # Plot the reconstruction values if requested
        plt.figure(figsize=(8, 6))
        plt.plot(num_elements, rec_x, marker='o')
        plt.ylabel("Reconstructed Cosine Similarity")
        plt.xlabel("Number of Principal Components Used")
        plt.title("Reconstruction Cosine Similarity vs. Number of Principal Components")
        plt.grid(True)
        plt.show()

    return data[:top_k]

def reconstruct_top_embedding_residual(data, embedding, mean, type, max_reconstr_score, top_k=10, approx=0.90):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
        - data (list): List of dictionaries containing details for each principal component of interest.
        - embedding: The embedding to reconstruct.
        - type: Type of embedding to reconstruct.
        - max_reconstr_score: Maximum achievable reconstruction score.
        - top_k: Number of top principal components to consider.
        - approx: Approximation threshold for the reconstruction score.

    Returns:
        - data: Data updated with top_k principal components of the given embeddings (if return_princ_comp is True).
    """

    # Store reconstruction values for plotting
    rec_x = []
    num_elements = []
    # Track Found heads and layers
    track_h_l = {}

    score = 0
    nr_pcs = 0
    query_curr = torch.zeros_like(embedding) 
    query_res = embedding
    data_best = []
    while nr_pcs != (top_k) and score / max_reconstr_score < approx:
        score *= 0
        max_score = torch.tensor(0)
        max_index = 0
        for count, entry in enumerate(data):
            [query_repres], _ = reconstruct_embeddings([data[count]], [query_res], [type], return_princ_comp=False)

            # Compute the current score:
            query_repres += mean
            embedding_dec = embedding + mean
            # Compute the current score: how well this partial reconstruction matches the original embedding
            score = embedding_dec @ query_repres.T
            
            if score.item() > max_score.item():
                max_score = score
                max_index = count
                if len(data_best) == nr_pcs + 1:
                    data_best[nr_pcs] = data[count]
                else:
                    data_best.append(data[count])

        # Save best PCs
        rec_x.append(max_score.item())
        nr_pcs += 1
        num_elements.append(nr_pcs)

        # Compute current query
        [query_repres], _ = reconstruct_embeddings([data[max_index]], [query_res], [type], return_princ_comp=False)

        query_curr += query_repres
        query_res -= query_repres

        # Add count of layer and head
        key = (data[max_index]["layer"], data[max_index]["head"])
        val = track_h_l.get(key, 0)
        track_h_l[key] = val
        track_h_l[key] += 1

        # Compute the current score:
        query_repres_norm = query_curr / query_curr.norm(dim=-1, keepdim=True)
        query_repres_norm += mean
        query_repres_norm /= query_repres_norm.norm(dim=-1, keepdim=True)
        embedding_dec = embedding + mean
        # Compute the current score: how well this partial reconstruction matches the original embedding
        score = embedding_dec @ query_repres_norm.T

        print(score)
    top_k = nr_pcs
    # Print information about how well the reconstruction performed
    percentage = (score / max_reconstr_score * 100).item()
    print(
        f"Reconstruction Quality Report:\n"
        f"- Maximum achievable reconstruction score: {max_reconstr_score.item():.4f}\n"
        f"- Used a total number of {len(track_h_l)} different heads\n"
        f"- Current reconstruction score: {score.item():.4f}\n"
        f"  This corresponds to {percentage:.2f}% of the maximum possible score.\n"
    )

    print(
        f"The reconstruction was performed using the top {top_k} principal component(s).\n "
        f"Increasing this number may improve the reconstruction score.\n\n"
    )

    # Plot the reconstruction values if requested
    plt.figure(figsize=(8, 6))
    plt.plot(num_elements, rec_x, marker='o')
    plt.ylabel("Reconstructed Cosine Similarity")
    plt.xlabel("Number of Principal Components Used")
    plt.title("Reconstruction Cosine Similarity vs. Number of Principal Components")
    plt.grid(True)
    plt.show()

    return data_best

def image_grid(images, rows, cols, labels=None, scores=None, scores_vis=None, figsize=None):
    """
    Display a collection of images arranged in a grid with optional labels and scores.
    
    Parameters:
        - images (list): List of image data (e.g., NumPy arrays or PIL Images) to display.
        - rows (int): Number of rows in the image grid.
        - cols (int): Number of columns in the image grid.
        - labels (list, optional): List of labels for each image. Defaults to None.
        - scores (list, optional): List of scores for each image, displayed alongside labels. Defaults to None.
        - scores_vis (list, optional): List of scores for each image, displayed alongside labels. Defaults to None.
        - figsize (tuple, optional): Size of the figure in inches as (width, height). If not provided, it is calculated based on rows and cols.
    
    Returns:
        - None
    """

    #TODO: reedit out
    labels = False
    scores = False
    scores_vis = False
    if figsize is None:
        figsize = (cols * 2, rows * 2)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            title = ''
            if labels:
                title += labels[i]
            if scores:
                title += f" ({scores[i]:.3f}) ({scores_vis[i]:.3f})"
            if title:
                ax.set_title(title, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_dataset(imagenet_path, transform, samples_per_class=3, tot_samples_per_class=50, seed=50, dataset="imagenet"):
    """
    Create a balanced subset of the ImageNet validation dataset for nearest-neighbor (NN) search.

    Args:
    - imagenet_path (str): Path to the root directory of the ImageNet dataset.
    - transform (callable): A function/transform to apply to the images in the dataset.
    - samples_per_class (int, optional): Number of samples to pick per class. Default is 3.
    - tot_samples_per_class (int, optional): Total number of samples per class considered as a block. Default is 50.
    - seed (int, optional): Random seed for reproducibility. Default is 50.

    Returns:
    - torch.utils.data.Subset: A subset of the ImageNet validation dataset containing the selected samples.
    """
    # Load the ImageNet validation dataset
    ds_vis = ImageNet(root=imagenet_path, split="val", transform=transform)

    # all_indices: a list of indices representing every sample in the validation set.
    all_indices = list(range(len(ds_vis)))
    nr_classes = len(ds_vis.classes)  # number of classes in the dataset

    # Set the random seed for reproducibility
    random.seed(seed)

    # For each class, randomly pick 'samples_per_class' samples from a block of 'tot_samples_per_class' samples.
    # This results in a balanced subset of classes.
    index = [random.sample(all_indices[x*tot_samples_per_class:(x + 1)*tot_samples_per_class], samples_per_class) 
             for x in range(nr_classes)]

    # Flatten the list of indices (since it's currently a list of lists)
    index = [x for xs in index for x in xs]

    # Create subsets of the dataset and visualization dataset using the selected indices
    if samples_per_class != tot_samples_per_class:
        print("Using subset of imagenet")
        ds_vis = torch.utils.data.Subset(ds_vis, index)
    return ds_vis


def create_dbs(scores_array_images, scores_array_texts, nr_top_imgs=20, nr_worst_imgs=20, nr_cont_imgs=20, text_query = None, max_reconstr_score = -1):
    """
    Create three sets of data containing top, worst, and continuous cosine similarity results 
    for images and corresponding text queries.

    Args:
    - data (iterable): The source dataset containing image and text samples.
    - scores_array_images (np.ndarray): Array of image scores, structured with 'score' and index values.
    - scores_array_texts (np.ndarray): Array of text scores, structured with 'score' and index values.
    - nr_top_imgs (int, optional): Number of top (highest cosine similarity) images to retrieve. Default is 20.
    - nr_worst_imgs (int, optional): Number of worst (lowest cosine similarity) images to retrieve. Default is 20.
    - nr_cont_imgs (int, optional): Number of continuous (interpolated similarity) images to retrieve. Default is 20.
    - text_query (str): The original query text for which cosine similarities are analyzed.
    - max_reconstr_score (float or torch.Tensor): The maximum reconstruction strength to display alongside results.

    Returns:
    - list of tuples: Each tuple contains:
        - db_images (np.ndarray): Selected image data based on scores.
        - db_texts (np.ndarray): Selected text data based on scores.
        - description (str): A string describing the type of retrieval (top, worst, or continuous).
        - num_samples (int): Number of samples in the current retrieval.
    """
    # Sort indices of scores
    sorted_scores_images = np.sort(scores_array_images, order=('score', 'score_vis'))

    # Top scores
    top_dbs_images = sorted_scores_images[-nr_top_imgs:][::-1]  # Get top `nr_top_imgs` elements, sorted descending

    # Worst scores
    worst_dbs_images = sorted_scores_images[:nr_worst_imgs]  # Get worst `nr_worst_imgs` elements, sorted ascending

    # Continuous scores (step-based interpolation)
    t = np.linspace(0, 1, num=nr_cont_imgs)  # Uniformly spaced in [0, 1]
    scaled_t = 0.5 * (1 - np.cos(np.pi * t))  # Cosine transformation for symmetry
    nonlinear_indices_images = (scaled_t * (len(sorted_scores_images) - 1)).astype(int)
    cont_dbs_images = sorted_scores_images[nonlinear_indices_images]


    if scores_array_texts is None:
        sorted_scores_texts = None
        top_dbs_texts = None
        cont_dbs_texts = None
        worst_dbs_texts = None
        nonlinear_indices_texts = None
    else:
        sorted_scores_texts = np.sort(scores_array_texts, order=('score', 'score_vis'))
        top_dbs_texts = sorted_scores_texts[-nr_top_imgs:][::-1]  # Get top `nr_top_imgs` elements, sorted descending
        worst_dbs_texts = sorted_scores_texts[:nr_worst_imgs]  # Get worst `nr_worst_imgs` elements, sorted ascending
        nonlinear_indices_texts = (scaled_t * (len(sorted_scores_texts) - 1)).astype(int)
        cont_dbs_texts = sorted_scores_texts[nonlinear_indices_texts]

    # Prepare three sets of results to display:
    # 1. Highest cosine similarity samples
    # 2. Most negative cosine similarity samples
    # 3. A continuous range of samples with intermediate similarity
    dbs = [
        (top_dbs_images, top_dbs_texts, 
         f"Printing highest cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if 
         text_query is not None else f"Printing highest cosine similarity",
         nr_top_imgs),
        (worst_dbs_images, worst_dbs_texts, 
         f"Printing most negative cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if 
         text_query is not None else f"Printing most negative cosine similarity",
         nr_worst_imgs),
        (cont_dbs_images, cont_dbs_texts, 
         f"Printing continuous cosine similarity with original query for text: '{text_query}' with strength {max_reconstr_score.item()}" if
         text_query is not None else f"Printing continuous cosine similarity",
         nr_cont_imgs)
    ]

    return dbs

def visualize_dbs_no_data(dbs, ds_vis, texts_str, classes):
    for db, db_text, display_text, length in dbs:
        images, labels, scores, scores_vis = [], [], [], []

        # Collect image samples, their class labels, and similarity scores
        for score, score_vis, image_index in db:
            images.append(ds_vis[image_index][0])  # Image data
            labels.append(classes[ds_vis[image_index][1]])  # Class label
            scores.append(score)
            scores_vis.append(score_vis)  # Cosine similarity score

        # Print the header describing the current subset
        print(display_text)

        # Prepare and display the corresponding texts with their scores
        output_rows = []

        if db_text is not None:
            for score, score_vis, text_index in db_text:
                output_rows.append([texts_str[text_index], score, score_vis])
            output_df = pd.DataFrame(output_rows, columns=["Text", "Cosine Similarity", "Correlation"])
            print(tabulate(output_df, headers='keys', tablefmt='psql'))

        # Display the images in a grid layout
        rows, cols = (length // 4, 4)  # Grid layout: 4 columns, rows derived from the number of images
        image_grid(images, rows, cols, labels=labels, scores=scores, scores_vis=scores_vis)

def visualize_dbs(data, dbs, ds_vis, texts_str, classes, text_query= None):
    """
    Visualize and analyze subsets of images and their associated text contributions 
    based on cosine similarity scores.

    Args:
    - data (iterable): Data containing principal components or details to analyze.
    - dbs (list of tuples): List of datasets to visualize, where each tuple contains:
        - db (list of tuples): Subset of images with scores and indices.
        - db_text (list of tuples): Subset of text entries with scores and indices.
        - display_text (str): Header text describing the current subset (e.g., "Top scores").
        - length (int): Number of samples in the current subset.
    - ds_vis (torch.utils.data.Dataset): Image dataset (e.g., ImageNet validation dataset).
    - texts_str (list of str): List of textual descriptions or queries corresponding to text indices.
    - text_query (str): The original query text being analyzed.
    - imagenet_classes (list of str): List of class names in the ImageNet dataset.

    Returns:
    - None
    """
    # Print a header to indicate which text query we are analyzing
    if text_query is not None:
        print(f"----Decomposition for text: '{text_query}'----")
        print_data(data, is_corr_present=True)

    else:
        print(f"----Decomposition of Principal Component----")
        print_data(data, is_corr_present=False)


    visualize_dbs_no_data(dbs, ds_vis, texts_str, classes)

def visualize_principal_component(
    layer, head, princ_comp, nr_top_imgs, nr_worst_imgs, nr_cont_imgs,
    attention_dataset, final_embeddings_images, final_embeddings_texts, 
    seed, data_path, texts_str, samples_per_class=3, dataset="imagenet",
    tot_samples_per_class=50,
    transform=visualization_preprocess
):
    """
    Visualize the top, worst, and continuous cosine similarity scores for images and texts 
    corresponding to a specific principal component of interest.

    Args:
    - layer (int): The attention layer to analyze.
    - head (int): The attention head within the specified layer.
    - princ_comp (int): The principal component index to visualize.
    - nr_top_imgs (int): Number of top (highest similarity) images to retrieve and display.
    - nr_worst_imgs (int): Number of worst (lowest similarity) images to retrieve and display.
    - nr_cont_imgs (int): Number of continuous (interpolated similarity) images to retrieve and display.
    - attention_dataset (str): Path to the JSON file containing attention data.
    - final_embeddings_images (torch.Tensor): Final image embeddings for similarity computation.
    - final_embeddings_texts (torch.Tensor): Final text embeddings for similarity computation.
    - seed (int): Random seed for reproducibility when creating the dataset.
    - imagenet_path (str): Path to the root directory of the ImageNet dataset.
    - texts_str (list of str): List of text descriptions corresponding to text embeddings.
    - transform (callable, optional): Transformation function to preprocess ImageNet images. 
                                      Defaults to `visualization_preprocess`.

    Returns:
    - None
    """
    # Retrieve and filter the principal component data
    data = get_data(attention_dataset, -1, skip_final=True)
    data = get_data_component(data, layer, head, princ_comp)
    

    # Dataset:
    if dataset == "imagenet":
        ds = ImageNet(root=data_path+"imagenet/", split="val", transform=transform)
    elif dataset == "binary_waterbirds":
        ds = BinaryWaterbirds(root=data_path+"waterbird_complete95_forest2water2/", split="test", transform=transform)
    elif dataset == "CIFAR100":
        ds = CIFAR100(
            root=data_path, download=True, train=False, transform=transform
        )
    elif dataset == "CIFAR10":
        ds = CIFAR10(
            root=data_path, download=True, train=False, transform=transform
        )
    else:
        ds = ImageFolder(root=data_path, transform=transform)

    # Depending
    ds_vis = dataset_subset(
        ds,
        samples_per_class=samples_per_class,
        tot_samples_per_class=tot_samples_per_class,  # or whatever you prefer
        seed=seed,
    )

    # final_embeddings_images covers the full dataset ds was computed on; if ds_vis is a subset,
    # restrict the embeddings to the same indices so img_index (below) stays aligned with ds_vis.
    if isinstance(ds_vis, torch.utils.data.Subset):
        final_embeddings_images = final_embeddings_images[ds_vis.indices]

    # Initialize arrays to store similarity scores for images and texts
    scores_array_images = np.empty(
        final_embeddings_images.shape[0], 
        dtype=[('score', 'f4'), ('score_vis', 'f4'), ('img_index', 'i4')]
    )
    scores_array_texts = np.empty(
        final_embeddings_texts.shape[0], 
        dtype=[('score', 'f4'), ('score_vis', 'f4'), ('txt_index', 'i4')]
    )

    # Create arrays of indices for referencing images and texts
    indexes_images = np.arange(0, final_embeddings_images.shape[0], 1)
    indexes_texts = np.arange(0, final_embeddings_texts.shape[0], 1)

    # Compute mean embeddings for centering
    mean_final_images = torch.mean(final_embeddings_images,  axis=0)
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    images_centered = final_embeddings_images# - mean_final_images
    texts_centered = final_embeddings_texts# - mean_final_texts

    # Normalize embeddings to unit norm

    # Compute cosine similarity scores with the specified principal component.
    # Images are scored the SAME way texts are (and the same way the PC's text labels were
    # assigned in svd_data_approx): project the full, unit-normalized shared-space embedding
    # onto the PC direction. Scoring the head's own contribution instead breaks this symmetry
    # and misaligns the top images from the text poles.
    vh = torch.tensor(data[0]["vh"])
    images_centered /= images_centered.norm(dim=-1, keepdim=True)

    scores_array_images["score_vis"] = ((images_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_images["score"] = ((images_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_images["img_index"] = indexes_images

    texts_centered /= texts_centered.norm(dim=-1, keepdim=True)

    scores_array_texts["score_vis"] = ((texts_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_texts["score"] = ((texts_centered @ vh.T)[:, princ_comp]).numpy()
    scores_array_texts["txt_index"] = indexes_texts

    # Create datasets for visualization
    dbs = create_dbs(
        scores_array_images, scores_array_texts, nr_top_imgs, 
        nr_worst_imgs, nr_cont_imgs, text_query=None, max_reconstr_score=-1
    )
    
    # Hardcoded visualizations
    nrs_dbs = [nr_top_imgs, nr_worst_imgs, nr_cont_imgs]
    dbs_new = []
    for i, db in enumerate(dbs):
        if nrs_dbs[i] == 0:
            continue
        dbs_new.append(db)
    
    classes = {
        'imagenet': imagenet_classes, 
        'CIFAR10': cifar_10_classes,
        'waterbirds': cub_classes, 
        'binary_waterbirds': waterbird_classes, 
        'cub': cub_classes}[dataset]
    
    # Visualize the results
    visualize_dbs(data, dbs_new, ds_vis, texts_str, classes, text_query=None)



@torch.no_grad()
def reconstruct_embeddings_proj(data, embeddings, types, device="cpu",return_princ_comp=False, plot=False, means=[]):
    """
    Reconstruct the embeddings using the principal components in data.
    Parameters:
    - data (list): List of dictionaries containing details for each principal component of interest.
    - embeddings (list): List containing the embeddings to reconstruct.
    - types (list): List of types of embeddings to reconstruct.
    - return_princ_comp (bool): Return the principal components of the given embeddings
    - plot (bool): Plot the reconstructed embeddings cosine similarity.
    - means (list): List of mean values for the embeddings.

    Returns:
    - list: Reconstructed embeddings NOT mean centered.
    - data: Data updated with principal components of the given embeddings (if return_princ_comp is True).
    """

    if len(embeddings) == 0 or len(types) != len(embeddings):
        assert False, "No embeddings to reconstruct or different lengths."

    # Initialize the reconstructed embeddings
    reconstructed_embeddings = [torch.zeros_like(embeddings[i].to(device)) for i in range(len(embeddings))]

    # Iterate over each principal component of interest
    all_pcs = torch.zeros((len(data), torch.tensor(data[0]["vh"]).shape[1])).to(device)
    #activationsdada
    for count, component in enumerate(data):
        # Retrieve projection matrices and mean values
        project_matrix = torch.tensor(component["project_matrix"]).to(device)
        vh = torch.tensor(component["vh"]).to(device)
        princ_comp = component["princ_comp"]
        s = component["strength_abs"]
        # Recover princ comp
        all_pcs[count] = vh[princ_comp, :]

    # Perform SVD of data matrix
    _, s, vh = torch.linalg.svd(all_pcs, full_matrices=False)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.999 # How much variance should cover the top princ_comps of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    print(f"The rank of the matrix is {rank}")
    for i in range(len(embeddings)):
        # Derive masking
        reconstructed_embeddings[i] = embeddings[i] @ vh.T @ vh


    return reconstructed_embeddings, data

def test_waterbird_preds(preds, labels, groups):
    # Count of water, water; water, land; land, water; land, land
    correct_count = [[0, 0], [0, 0]]
    tot_count = [[0, 0], [0, 0]]

    for pred, label, group in zip(preds, labels, groups):
        # Increment the total count for the group
        tot_count[label][group] += 1

        # Increment the correct count if the prediction is correct
        if pred:
            correct_count[label][group] += 1

    print("Accuracy landbird with landbird background:", 100* correct_count[0][0] / tot_count[0][0])
    print("Accuracy landbird with water background:", 100* correct_count[0][1] / tot_count[0][1])
    print("Accuracy waterbird with land background:", 100* correct_count[1][0] / tot_count[1][0])
    print("Accuracy waterbird with water background:", 100* correct_count[1][1] / tot_count[1][1])

    print("Totoal accuracy landbird:", 100* (correct_count[0][0] + correct_count[0][1]) / (tot_count[0][0] + tot_count[0][1]))
    print("Totoal accuracy waterbird:", 100* (correct_count[1][0] + correct_count[1][1]) / (tot_count[1][0] + tot_count[1][1]))
    print("Total accuracy overall", 100* (correct_count[0][0] + correct_count[0][1] + correct_count[1][0] + correct_count[1][1]) / (tot_count[0][0] + tot_count[0][1] + tot_count[1][0] + tot_count[1][1]))

    return 100*min(correct_count[0][0] / tot_count[0][0], correct_count[0][1] / tot_count[0][1], correct_count[1][0] / tot_count[1][0], correct_count[1][1] / tot_count[1][1])

def test_accuracy(prediction, labels, label="Classifier"):
    """
    Calculate the accuracy of the model's predictions.
    """
    accuracy, indexes = accuracy_correct(prediction, labels)
    accuracy_pred = accuracy[0] * 100
    print(f"For the approach {label}, the accuracy is: {accuracy_pred:3f}%")
    return accuracy_pred, indexes[0]

def print_wrong_elements_label(indexes_1, label, subset_dim, text="wrong"):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    idx_label = [idx for idx, x in enumerate(imagenet_classes) if x == label][0]
    count = 0
    for idx in range(idx_label*subset_dim, idx_label*subset_dim + subset_dim):
        if not indexes_1[idx]:
            count += 1
        
    # Print the result
    print(f"The {label} {text} elements labels are: {count}")

    return count

def print_tot_wrong_elements_label(predictions, label):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    curr_preds = predictions.argmax(dim=1)
    label_idx = imagenet_classes.index(label)
    count = 0
    for pred in curr_preds:
        if pred == label_idx:
            count +=1
        
    # Print the result
    print(f"In total, for the {label} label, there are {count} wrong elements.")


    return count
def print_diff_elements(indexes_1, indexes_2, subset_dim):
    # TODO: Hardcoded for ImageNet
    # Retrieve the labels of the dataset. 
    # This is hardcoded for ImageNet where nr_classes is the number of classes (usually 1000).
    nr_samples = torch.arange(1000)
    classes_indexes = nr_samples.repeat_interleave(subset_dim)
    class_labels = np.array([imagenet_classes[i] for i in classes_indexes])
    # Determine which elements differ between the two reconstructions
    wrong_elements = np.array(~(indexes_1 == indexes_2))
    
    print(f"Number of elements with different results between the two reconstruction methods: {len(class_labels[wrong_elements])}")
    # Track occurrences
    label_count = defaultdict(int)
    output_set = set()

    # Iterate through mask and labels
    for idx, is_wrong in enumerate(wrong_elements):
        if is_wrong:
            label = class_labels[idx]
            label_count[label] += 1

    # Sort the set by nr_of_prev_occurrences in descending order
    sorted_output = sorted(label_count.items(), key=lambda x: x[1], reverse=True)

    # Print the result
    print(f"The different elements labels are: {sorted_output}")

def print_wrong_elements(indexes_1, labels, classes, text="wrong"):
    # Retrieve the labels of the dataset. 
    class_labels = classes
    
    print(f"Number of elements with {text} results between the two reconstruction methods: {len(labels[indexes_1])}")
    # Track occurrences
    label_count = defaultdict(int)

    # Iterate through mask and labels
    for idx, label in zip(indexes_1, labels):
        if not idx:
            class_label = class_labels[label]
            label_count[class_label] += 1

    # Sort the set by nr_of_prev_occurrences in descending order
    sorted_output = sorted(label_count.items(), key=lambda x: x[1], reverse=True)

    # Print the result
    print(f"The {text} elements labels are: {sorted_output}")
    return sorted_output


def print_correct_elements(indexes_1, labels, classes, text="correct"):
    return print_wrong_elements(~indexes_1, labels, classes, text)


def save_parameters_proof_of_concept(worst_class_acc, total_accuracy, worst_class_nr_pcs, 
                    baseline_worst, baseline_acc, model_name, concept_nr, classes_):
    # Convert any numpy arrays to lists, if needed:
    def to_serializable(x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    data = {
        "worst_class_acc": to_serializable(worst_class_acc),
        "total_accuracy": to_serializable(total_accuracy),
        "worst_class_nr_pcs": to_serializable(worst_class_nr_pcs),
        "baseline_worst": baseline_worst,
        "baseline_acc": baseline_acc,
        "model_name": model_name,
        "concept_nr": concept_nr,
        "classes_": classes_  # Assumes classes_ is already JSON-serializable (e.g., a list)
    }

    # Create a meaningful filename using model_name and concept_nr
    file_name = f"params_{model_name}_proof_of_concept_{concept_nr}.json"
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Parameters saved to '{file_name}'.")

# -----------------------------------------------------------------------------
# Function to load parameters from a JSON file
# -----------------------------------------------------------------------------
def load_parameters(file_name):
    with open(file_name, "r") as f:
        data = json.load(f)
    return data


def plot_proof_of_concept(worst_class_acc, total_accuracy, worst_class_nr_pcs, baseline_worst, baseline_acc, model_name, concept_nr, classes_):
    # ---------------------------------------
    # PLOTTING SECTION for WORST-CLASS ACCURACY
    # ---------------------------------------
    # -------------------------------
    # COMPUTE PARETO OPTIMAL CANDIDATE
    # -------------------------------
    # A candidate is one where both total accuracy and worst-class accuracy exceed their baselines.
    # Among these, we select the one with the highest worst-class accuracy.
    # If there is a tie, we pick the candidate with the higher total accuracy.
    total_accuracy_arr = np.array(total_accuracy)
    worst_class_acc_arr = np.array(worst_class_acc)
    worst_class_nr_pcs_arr = np.array(worst_class_nr_pcs)

    candidates_mask = (total_accuracy_arr > baseline_acc) & (worst_class_acc_arr > baseline_worst)
    if np.any(candidates_mask):
        candidate_indices = np.where(candidates_mask)[0]
        best_idx = candidate_indices[0]
        for idx in candidate_indices:
            if worst_class_acc_arr[idx] > worst_class_acc_arr[best_idx]:
                best_idx = idx
            elif worst_class_acc_arr[idx] == worst_class_acc_arr[best_idx] and total_accuracy_arr[idx] > total_accuracy_arr[best_idx]:
                best_idx = idx
        pareto_pcs = worst_class_nr_pcs_arr[best_idx]
        pareto_worst = worst_class_acc_arr[best_idx]
        pareto_total = total_accuracy_arr[best_idx]
    else:
        pareto_pcs = None

    # -------------------------------
    # PLOTTING SECTION for WORST-CLASS ACCURACY
    # -------------------------------
    # (Only applicable if classes_ == waterbird_classes)
    if classes_ == waterbird_classes:
        # Recompute max values for worst-class accuracy for clarity
        max_worst_acc = max(worst_class_acc)
        pcs_per_class_max_worst_acc = worst_class_nr_pcs[np.argmax(worst_class_acc)]

        plt.figure(figsize=(6, 4))
        # Plot the worst-class accuracy curve
        plt.plot(
            worst_class_nr_pcs, worst_class_acc,
            color='blue',
            linestyle='-',
            label='Worst-Class Accuracy'
        )
        # Plot baseline worst-class accuracy
        plt.axhline(
            y=baseline_worst,
            color='gray',
            linestyle='--',
            linewidth=2,
            label=f'Baseline Worst-Class Acc = {baseline_worst:.2f}'
        )
        # Highlight the maximum worst-class accuracy with horizontal and vertical lines
        plt.axhline(
            y=max_worst_acc,
            color='blue',
            linestyle=':',
            linewidth=2
        )
        plt.axvline(
            x=pcs_per_class_max_worst_acc,
            color='blue',
            linestyle=':',
            linewidth=2
        )
        # --- Add Vertical Line for Pareto Optimal Solution (if found) ---
        if pareto_pcs is not None:
            plt.axvline(
                x=pareto_pcs,
                color='green',
                linestyle='--',
                linewidth=2
            )
            pareto_line_legend = mlines.Line2D(
                [],
                [],
                color='green',
                linestyle='--',
                linewidth=2,
                label=f'Pareto Optimal: Worst Acc = {pareto_worst:.2f}, Total Acc = {pareto_total:.2f} at PCs={pareto_pcs}'
            )
        else:
            pareto_line_legend = None

        # Custom legend entry for max worst-class accuracy
        max_line_legend = mlines.Line2D(
            [],
            [],
            color='blue',
            linestyle=':',
            linewidth=2,
            label=f'Max Worst-Class Acc = {max_worst_acc:.2f} at PCs={pcs_per_class_max_worst_acc}\n(Worst total accuracy is {total_accuracy[np.argmax(worst_class_acc)]:.2f})'
        )

        # Get current legend handles and append our custom entries
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(max_line_legend)
        labels.append(max_line_legend.get_label())
        if pareto_line_legend is not None:
            handles.append(pareto_line_legend)
            labels.append(pareto_line_legend.get_label())

        plt.xlabel('Number of PCs per Class')
        plt.ylabel('Worst Accuracy')
        plt.title('Worst-Class Accuracy vs. Number of PCs per Class')
        plt.grid(True)
        plt.legend(handles, labels)
        plt.tight_layout()
        plt.savefig(f"plt_1worst_{model_name}.pdf", bbox_inches='tight', format='pdf')
        plt.show()

        print(f"Max worst accuracy of {max_worst_acc:.2f} found at {pcs_per_class_max_worst_acc} PCs/class.")
        if pareto_pcs is not None:
            print(f"Pareto optimal solution at {pareto_pcs} PCs/class with Worst Acc = {pareto_worst:.2f} and Total Acc = {pareto_total:.2f}.")

    # -------------------------------
    # PLOTTING SECTION for TOTAL ACCURACY
    # -------------------------------
    # Compute max total accuracy details
    max_total_acc = max(total_accuracy)
    pcs_for_max_total_acc = worst_class_nr_pcs[np.argmax(total_accuracy)]

    plt.figure(figsize=(6, 4))
    # Plot the total accuracy curve
    plt.plot(
        worst_class_nr_pcs, total_accuracy,
        linestyle='-', color='orange',
        label='Total Accuracy'
    )
    # Plot baseline total accuracy
    plt.axhline(
        y=baseline_acc,
        color='gray',
        linestyle='--',
        linewidth=2,
        label=f'Baseline Total Acc = {baseline_acc:.2f}'
    )
    # Highlight the maximum total accuracy with horizontal and vertical lines
    plt.axhline(
        y=max_total_acc,
        color='orange',
        linestyle=':',
        linewidth=2
    )
    plt.axvline(
        x=pcs_for_max_total_acc,
        color='orange',
        linestyle=':',
        linewidth=2
    )
    # --- Add Vertical Line for Pareto Optimal Solution (if found) ---
    if pareto_pcs is not None:
        plt.axvline(
            x=pareto_pcs,
            color='green',
            linestyle='--',
            linewidth=2
        )
        pareto_line_legend_2 = mlines.Line2D(
            [],
            [],
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Pareto Optimal: Worst Acc = {pareto_worst:.2f}, Total Acc = {pareto_total:.2f} at PCs={pareto_pcs}'
        )
    else:
        pareto_line_legend_2 = None

    # Custom legend entry for max total accuracy
    max_line_legend_2 = mlines.Line2D(
        [],
        [],
        color='orange',
        linestyle=':',
        linewidth=2,
        label=f'Max Total Acc = {max_total_acc:.2f} at PCs={pcs_for_max_total_acc}\n(Worst class accuracy is {worst_class_acc[np.argmax(total_accuracy)]:.2f})'
    )

    handles2, labels2 = plt.gca().get_legend_handles_labels()
    handles2.append(max_line_legend_2)
    labels2.append(max_line_legend_2.get_label())
    if pareto_line_legend_2 is not None:
        handles2.append(pareto_line_legend_2)
        labels2.append(pareto_line_legend_2.get_label())

    plt.xlabel('Number of PCs per Class')
    plt.ylabel('Total Accuracy (Avg)')
    plt.title('Total Accuracy vs. Number of PCs per Class')
    plt.grid(True)
    plt.legend(handles2, labels2)
    plt.tight_layout()
    plt.savefig(f"plt_1_acc_{model_name}.pdf", bbox_inches='tight', format='pdf')
    plt.show()

    print(f"Max total accuracy of {max_total_acc:.2f} found at {pcs_for_max_total_acc}")


def proof_concept_1_pcs_all(classifier_, attns_, mlps_, labels_, classes_, model_name, background_groups_, final_embeddings_texts, attention_dataset, nr_layers_, nr_heads_, num_last_layers_, pcs_per_class_start=1, pcs_per_class_end=2000, pcs_per_class_step=10, max_pcs_per_head=-1, random=False):   
    # Embeddings classes
    class_embeddings = classifier_.T  # M x D
    # Baseline accuracy computation:
    baseline = attns_.sum(axis=(1, 2)) + mlps_.sum(axis=1)
    baseline_acc, idxs = test_accuracy(baseline @ classifier_, labels_, label="Baseline")
    print_correct_elements(idxs, labels_, classes_)  

    if classes_ == waterbird_classes:
        baseline_worst = test_waterbird_preds(idxs, labels_, background_groups_)
    else:
        baseline_worst = None
    # Using the knwoledge of which class are we predicting wrongly, give more or less weight to pcs per class
    # Reconstruct embeddings for each class label

    # Get mean of data and texts
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    classes_centered = class_embeddings - mean_final_texts.unsqueeze(0)

    sorted_data = []
    for text_idx in range(classes_centered.shape[0]):
        # Perform query system on entry
        concept_i_centered = classes_centered[text_idx, :].unsqueeze(0)

        data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)

        _, data_abs = reconstruct_embeddings(
            data, 
            [concept_i_centered], 
            ["text"], 
            return_princ_comp=True, 
            plot=False, 
            means=[mean_final_texts],
        )

        # Extract relevant details from the top k entries
        data_pcs = sort_data_by(data_abs, "correlation_princ_comp_abs", descending=True)
        # Derive nr_pcs_per_class
        sorted_data.append(data_pcs)

    max_worst_acc = 0

    worst_class_acc = []
    worst_class_nr_pcs = []

    total_accuracy = []  # Will store the average accuracy across all text_idx for each pcs_per_class

    for pcs_per_class in range(pcs_per_class_start, pcs_per_class_end, pcs_per_class_step):
        entries = []

        # Collect top_k_entries for each concept
        for text_idx in range(classes_centered.shape[0]):
            # Retrieve data
            data_pcs = sorted_data[text_idx]
            top_k_entries = top_data(data_pcs, pcs_per_class)
            print(f"Currently processing label {text_idx} with nr_pcs_per_class: {pcs_per_class}")
            entries += top_k_entries

        # Remove duplicates
        entries_set = []
        entries_meta = []
        for entry in entries:
            layer = entry["layer"]
            head = entry["head"]
            princ_comp = entry["princ_comp"]
            if (layer, head, princ_comp) not in entries_meta:
                entries_meta.append((layer, head, princ_comp))
                entries_set.append(entry)

        print(f"Total number of unique entries: {len(entries_set)}")

        # If `random` is True, randomly pick PCs instead of the actual top_k
        if random:
            data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)
            entries_set = random_pcs(data, pcs_per_class * len(classes_))

        # Reconstruct final_embeddings_images
        reconstructed_images = reconstruct_all_embeddings_mean_ablation_pcs(
            entries_set,
            mlps_,
            attns_,
            attns_,
            nr_layers_,
            nr_heads_,
            num_last_layers_,
            ratio=-1,
            mean_ablate_all=False
        )

        reconstructed_images /= reconstructed_images.norm(dim=-1, keepdim=True)
        predictions = reconstructed_images @ classifier_

        # Evaluate across all text_idx
        # (If you have different labels per text_idx, adapt accordingly.)
        # For simplicity, assume `test_accuracy` returns (acc, idxs) for the entire set.
        acc, idxs = test_accuracy(predictions, labels_, label="All classes combined")
        total_accuracy.append(acc)

        # If you need to evaluate `acc` separately for each text_idx, do so in the loop above.
        # Then store the average or final result below.

        # You might also be printing correctness here:
        print_correct_elements(idxs, labels_, classes_)

        # Worst-class accuracy for Waterbirds, if applicable
        if classes_ == waterbird_classes:
            curr_worst_acc = test_waterbird_preds(idxs, labels_, background_groups_)
            if curr_worst_acc > max_worst_acc:
                max_worst_acc = curr_worst_acc

            worst_class_acc.append(curr_worst_acc)
            worst_class_nr_pcs.append(pcs_per_class)

    if classes_ == waterbird_classes:
        concept_nr = "1"
        # Save parameters to a JSON file
        save_parameters_proof_of_concept(worst_class_acc, total_accuracy, worst_class_nr_pcs,
                        baseline_worst, baseline_acc, model_name, concept_nr, classes_)

        # Suppose later (or in another script) you want to recover the parameters:
        file_to_load = f"params_{model_name}_proof_of_concept_{concept_nr}.json"
        params = load_parameters(file_to_load)
        plot_proof_of_concept(**params)


def proof_concept_2_remove(classifier_, attns_, mlps_, labels_, classes_, model_name, model, tokenizer, device, background_groups_, final_embeddings_texts, attention_dataset, nr_layers_, nr_heads_, num_last_layers_, concepts_to_remove = ["water background", "land background"], pcs_per_class_start=1, pcs_per_class_end=1500, pcs_per_class_step=10, max_pcs_per_head=-1, random=False):
    # Derive embedding:
    for k, concept in enumerate(concepts_to_remove):
        # Retrieve an embedding
        with torch.no_grad():
            # If querying by text, define a text prompt and encode it into an embedding
            # Tokenize the text query and move it to the device (GPU/CPU)
            text_query_token = tokenizer(concept).to(device)  
            # Encode the tokenized text into a normalized embedding
            topic_emb = model.encode_text(text_query_token, normalize=True)
            if k == 0:
                concepts_emb = torch.zeros(len(concepts_to_remove), topic_emb.shape[-1], device=device)
            concepts_emb[k] = topic_emb

    # Print baseline accuracy
    # Baseline accuracy computation:
    baseline = attns_.sum(axis=(1, 2)) + mlps_.sum(axis=1)
    baseline_acc, idxs = test_accuracy(baseline @ classifier_, labels_, label="Baseline")
    if classes_ == waterbird_classes:
        baseline_worst = test_waterbird_preds(idxs, labels_, background_groups_)
    else:
        baseline_worst = None
    # Reconstruct embeddings for each class label

    # Get mean of data and texts
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    concepts_centered = concepts_emb - mean_final_texts.unsqueeze(0)

    sorted_data = []
    for text_idx in range(concepts_centered.shape[0]):
        # Perform query system on entry
        concept_i_centered = concepts_centered[text_idx, :].unsqueeze(0)

        data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)

        _, data_abs = reconstruct_embeddings(
            data, 
            [concept_i_centered], 
            ["text"], 
            return_princ_comp=True, 
            plot=False, 
            means=[mean_final_texts],
        )

        # Extract relevant details from the top k entries
        data_pcs = sort_data_by(data_abs, "correlation_princ_comp_abs", descending=True)
        # Derive nr_pcs_per_class
        sorted_data.append(data_pcs)

    max_worst_acc = 0

    worst_class_acc = []
    worst_class_nr_pcs = []

    total_accuracy = []  # Will store the average accuracy across all text_idx for each pcs_per_class

    for pcs_per_class in range(pcs_per_class_start, pcs_per_class_end, pcs_per_class_step):
        entries = []

        # Collect top_k_entries for each concept
        for text_idx in range(concepts_centered.shape[0]):
            # Retrieve data
            data_pcs = sorted_data[text_idx]
            top_k_entries = top_data(data_pcs, pcs_per_class)
            print(f"Currently processing label: {concepts_to_remove[text_idx]} with nr_pcs_per_class: {pcs_per_class}")
            entries += top_k_entries

        # Remove duplicates
        entries_set = []
        entries_meta = []
        for entry in entries:
            layer = entry["layer"]
            head = entry["head"]
            princ_comp = entry["princ_comp"]
            if (layer, head, princ_comp) not in entries_meta:
                entries_meta.append((layer, head, princ_comp))
                entries_set.append(entry)

        print(f"Total number of unique entries: {len(entries_set)}")

        # Extract other components
        top_k_other_details = get_remaining_pcs(data, entries_set)

        # If `random` is True, randomly pick PCs instead of the actual top_k
        if random:
            data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)
            top_k_other_details = random_pcs(data, pcs_per_class * len(classes_))

        # Reconstruct final_embeddings_images
        reconstructed_images = reconstruct_all_embeddings_mean_ablation_pcs(
            top_k_other_details,
            mlps_,
            attns_,
            attns_,
            nr_layers_,
            nr_heads_,
            num_last_layers_,
            ratio=-1,
            mean_ablate_all=False
        )

        reconstructed_images /= reconstructed_images.norm(dim=-1, keepdim=True)
        predictions = reconstructed_images @ classifier_

        # Evaluate across all text_idx
        # (If you have different labels per text_idx, adapt accordingly.)
        # For simplicity, assume `test_accuracy` returns (acc, idxs) for the entire set.
        acc, idxs = test_accuracy(predictions, labels_, label="All classes combined")
        total_accuracy.append(acc)

        # If you need to evaluate `acc` separately for each text_idx, do so in the loop above.
        # Then store the average or final result below.

        # You might also be printing correctness here:
        print_correct_elements(idxs, labels_, classes_)

        # Worst-class accuracy for Waterbirds, if applicable
        if classes_ == waterbird_classes:
            curr_worst_acc = test_waterbird_preds(idxs, labels_, background_groups_)
            if curr_worst_acc > max_worst_acc:
                max_worst_acc = curr_worst_acc
                pcs_per_class_max_worst_acc = pcs_per_class

            worst_class_acc.append(curr_worst_acc)
            worst_class_nr_pcs.append(pcs_per_class)
    
    if classes_ == waterbird_classes:
        concept_nr = "2"
        # Save parameters to a JSON file
        save_parameters_proof_of_concept(worst_class_acc, total_accuracy, worst_class_nr_pcs,
                        baseline_worst, baseline_acc, model_name, concept_nr, classes_)

        # Suppose later (or in another script) you want to recover the parameters:
        file_to_load = f"params_{model_name}_proof_of_concept_{concept_nr}.json"
        params = load_parameters(file_to_load)
        plot_proof_of_concept(**params)


def proof_concept_3_add(classifier_, attns_, mlps_, labels_, classes_, model_name, model, tokenizer, device, background_groups_, final_embeddings_texts, attention_dataset, nr_layers_, nr_heads_, num_last_layers_, concepts_to_add = ["feet shape", "beak shape"], pcs_per_class_start=1, pcs_per_class_end=1500, pcs_per_class_step=10, max_pcs_per_head=-1, random=False):
    class_embeddings = classifier_.T  # M x D
    # Derive embedding:
    for k, concept in enumerate(concepts_to_add):
        # Retrieve an embedding
        with torch.no_grad():
            # If querying by text, define a text prompt and encode it into an embedding
            # Tokenize the text query and move it to the device (GPU/CPU)
            text_query_token = tokenizer(concept).to(device)  
            # Encode the tokenized text into a normalized embedding
            topic_emb = model.encode_text(text_query_token, normalize=True)
            if k == 0:
                concepts_emb = torch.zeros(len(concepts_to_add), topic_emb.shape[-1], device=device)
            concepts_emb[k] = topic_emb

    # Print baseline accuracy
    # Baseline accuracy computation:
    baseline = attns_.sum(axis=(1, 2)) + mlps_.sum(axis=1)
    baseline_acc, idxs = test_accuracy(baseline @ classifier_, labels_, label="Baseline")
    if classes_ == waterbird_classes:
        baseline_worst = test_waterbird_preds(idxs, labels_, background_groups_)
    else:
        baseline_worst = None
    # Reconstruct embeddings for each class label

    # Get mean of data and texts
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    concepts_centered = concepts_emb - mean_final_texts.unsqueeze(0)

    sorted_data = []
    for text_idx in range(concepts_centered.shape[0]):
        # Perform query system on entry
        concept_i_centered = concepts_centered[text_idx, :].unsqueeze(0)

        data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)

        _, data_abs = reconstruct_embeddings(
            data, 
            [concept_i_centered], 
            ["text"], 
            return_princ_comp=True, 
            plot=False, 
            means=[mean_final_texts],
        )

        # Extract relevant details from the top k entries
        data_pcs = sort_data_by(data_abs, "correlation_princ_comp_abs", descending=True)
        # Derive nr_pcs_per_class
        sorted_data.append(data_pcs)

    max_worst_acc = 0

    worst_class_acc = []
    worst_class_nr_pcs = []

    total_accuracy = []  # Will store the average accuracy across all text_idx for each pcs_per_class

    for pcs_per_class in range(pcs_per_class_start, pcs_per_class_end, pcs_per_class_step):
        entries = []

        # Collect top_k_entries for each concept
        for text_idx in range(concepts_centered.shape[0]):
            # Retrieve data
            data_pcs = sorted_data[text_idx]
            top_k_entries = top_data(data_pcs, pcs_per_class)
            print(f"Currently processing label: {concepts_to_add[text_idx]} with nr_pcs_per_class: {pcs_per_class}")
            entries += top_k_entries

        # Remove duplicates
        entries_set = []
        entries_meta = []
        for entry in entries:
            layer = entry["layer"]
            head = entry["head"]
            princ_comp = entry["princ_comp"]
            if (layer, head, princ_comp) not in entries_meta:
                entries_meta.append((layer, head, princ_comp))
                entries_set.append(entry)

        print(f"Total number of unique entries: {len(entries_set)}")

        # If `random` is True, randomly pick PCs instead of the actual top_k
        if random:
            data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)
            entries_set = random_pcs(data, pcs_per_class * len(classes_))

        # Reconstruct final_embeddings_images
        reconstructed_images = reconstruct_all_embeddings_mean_ablation_pcs(
            entries_set,
            mlps_,
            attns_,
            attns_,
            nr_layers_,
            nr_heads_,
            num_last_layers_,
            ratio=-1,
            mean_ablate_all=False
        )

        reconstructed_images /= reconstructed_images.norm(dim=-1, keepdim=True)
        predictions = reconstructed_images @ classifier_

        # Evaluate across all text_idx
        # (If you have different labels per text_idx, adapt accordingly.)
        # For simplicity, assume `test_accuracy` returns (acc, idxs) for the entire set.
        acc, idxs = test_accuracy(predictions, labels_, label="All classes combined")
        total_accuracy.append(acc)

        # If you need to evaluate `acc` separately for each text_idx, do so in the loop above.
        # Then store the average or final result below.

        # You might also be printing correctness here:
        print_correct_elements(idxs, labels_, classes_)

        # Worst-class accuracy for Waterbirds, if applicable
        if classes_ == waterbird_classes:
            curr_worst_acc = test_waterbird_preds(idxs, labels_, background_groups_)
            if curr_worst_acc > max_worst_acc:
                max_worst_acc = curr_worst_acc

            worst_class_acc.append(curr_worst_acc)
            worst_class_nr_pcs.append(pcs_per_class)

    if classes_ == waterbird_classes:
        concept_nr = "3"
        # Save parameters to a JSON file
        save_parameters_proof_of_concept(worst_class_acc, total_accuracy, worst_class_nr_pcs,
                        baseline_worst, baseline_acc, model_name, concept_nr, classes_)

        # Suppose later (or in another script) you want to recover the parameters:
        file_to_load = f"params_{model_name}_proof_of_concept_{concept_nr}.json"
        params = load_parameters(file_to_load)
        plot_proof_of_concept(**params)

def proof_concept_4_pcs_single(classifier_, attns_, mlps_, labels_, classes_, model_name, background_groups_, final_embeddings_texts, final_embeddings_images, subset_dim, attention_dataset, nr_layers_, nr_heads_, num_last_layers_, pcs_per_class_start=1, pcs_per_class_end=2000, pcs_per_class_step=10, max_pcs_per_head=-1, random=False):   
    class_embeddings = classifier_.T  # M x D

    # Print baseline accuracy
    # Baseline accuracy computation:
    baseline = attns_.sum(axis=(1, 2)) + mlps_.sum(axis=1)
    baseline_acc, idxs = test_accuracy(baseline @ classifier_, labels_, label="Baseline")
    if classes_ == waterbird_classes:
        baseline_worst = test_waterbird_preds(idxs, labels_, background_groups_)
    else:
        baseline_worst = None
    # Reconstruct embeddings for each class label

    # Get mean of data and texts
    mean_final_images = torch.mean(final_embeddings_images, axis=0)
    mean_final_texts = torch.mean(final_embeddings_texts, axis=0)

    classes_centered = class_embeddings - mean_final_texts.unsqueeze(0)

    # Initialize a (num_images x 2) array to track:
    #   [best_score_so_far, class_index_for_that_score]

    sorted_data = []
    for text_idx in range(classes_centered.shape[0]):
        # Perform query system on entry
        concept_i_centered = classes_centered[text_idx, :].unsqueeze(0)

        data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)

        _, data_abs = reconstruct_embeddings(
            data, 
            [concept_i_centered], 
            ["text"], 
            return_princ_comp=True, 
            plot=False, 
            means=[mean_final_texts],
        )

        # Extract relevant details from the top k entries
        data_pcs = sort_data_by(data_abs, "correlation_princ_comp_abs", descending=True)
        # Derive nr_pcs_per_class
        sorted_data.append(data_pcs)

    max_worst_acc = 0

    worst_class_acc = []
    worst_class_nr_pcs = []

    total_accuracy = []  # Will store the average accuracy across all text_idx for each pcs_per_class

    for pcs_per_class in range(pcs_per_class_start, pcs_per_class_end, pcs_per_class_step):
        all_preds = torch.zeros((final_embeddings_images.shape[0], 2), dtype=torch.double)

        for text_idx in range(classes_centered.shape[0]):
            # Retrieve data
            data_pcs = sorted_data[text_idx]
            top_k_entries = top_data(data_pcs, pcs_per_class)
            # If `random` is True, randomly pick PCs instead of the actual top_k
            if random:
                data = get_data(attention_dataset, max_pcs_per_head, skip_final=True)
                top_k_entries = random_pcs(data, pcs_per_class * len(classes_))
                
            # Reconstruct final_embeddings_images
            reconstructed_images = reconstruct_all_embeddings_mean_ablation_pcs(
                top_k_entries,
                mlps_,
                attns_, 
                attns_,
                nr_layers_,
                nr_heads_,
                num_last_layers_,
                ratio=-1,
                mean_ablate_all=False
            )

            reconstructed_images /= reconstructed_images.norm(dim=-1, keepdim=True)
            predictions = reconstructed_images @ class_embeddings[text_idx, :].T #class_embeddings[text_idx, :].T
            # Update "best so far" scores in all_preds
            best_vals_this_round = predictions
            improved_mask = best_vals_this_round > all_preds[:, 0]

            best_idxs_this_round = torch.full_like(all_preds[:, 1], fill_value=text_idx)
            all_preds[improved_mask, 0] = best_vals_this_round[improved_mask].double()
            all_preds[improved_mask, 1] = best_idxs_this_round[improved_mask].double()

            # Optionally, check accuracy for the current text_idx predictions
            acc, idxs = test_accuracy(predictions.unsqueeze(-1), labels_, label=f"{classes_[text_idx]}")
            print_correct_elements(idxs, labels_, classes_)

            # Build a fictitious one-hot matrix from all_preds
            num_images = final_embeddings_images.shape[0]
            num_classes = classifier_.shape[1]  # Typically M x D => M classes => classifier_.shape[1] is #classes

            # Convert the best class index to a LongTensor
            best_class_idxs = all_preds[:, 1].long()

            # Create zero matrix [num_images, num_classes]
            fictitious_preds = torch.zeros((num_images, num_classes), device=best_class_idxs.device)

            # Fill 1.0 in the best predicted class for each image
            fictitious_preds[torch.arange(num_images), best_class_idxs] = 1.0

            # Test accuracy on these "hard" predictions
            acc_best, idxs_best = test_accuracy(fictitious_preds, labels_, label="Best So Far (One-Hot)")
            if text_idx == len(classes_centered) - 1:
                total_accuracy.append(acc_best)

            sorted_output = print_correct_elements(idxs_best, labels_, classes_)
            if classes_ == waterbird_classes:
                curr_worst_acc = test_waterbird_preds(idxs_best, labels_, background_groups_)
                if text_idx == len(classes_centered) - 1:
                    if curr_worst_acc > max_worst_acc:
                        max_worst_acc = curr_worst_acc
                        pcs_per_class_max_worst_acc = pcs_per_class

                    worst_class_acc.append(curr_worst_acc)
                    worst_class_nr_pcs.append(pcs_per_class)

            # Print overall accuracy so far
            tot_sum = 0
            for _, el_nr in sorted_output:
                tot_sum += el_nr
            if subset_dim == None:
                print(f"Tot accuracy so far is {tot_sum/len(labels_)}")

            else:
                print(f"Tot accuracy so far is {tot_sum/((text_idx + 1) * subset_dim)}")
    
    if classes_ == waterbird_classes:    
        concept_nr = "4"
        # Save parameters to a JSON file
        save_parameters_proof_of_concept(worst_class_acc, total_accuracy, worst_class_nr_pcs,
                        baseline_worst, baseline_acc, model_name, concept_nr, classes_)

        # Suppose later (or in another script) you want to recover the parameters:
        file_to_load = f"params_{model_name}_proof_of_concept_{concept_nr}.json"
        params = load_parameters(file_to_load)
        plot_proof_of_concept(**params)

# --------------------------------------------------------------------------- #
# Transfer / cross-dataset completeness
# --------------------------------------------------------------------------- #
def transfer_completeness(
    input_dir, model, seed, tower, fit_dataset, eval_datasets,
    components="all", textset="top_1500_nouns_5_sentences_imagenet_clean",
    text_dir="./utils/text_descriptions", image_set="self", probe_name="imagenet",
    num_of_last_layers=999, text_per_pc=5, max_text=999, device="cpu",
):
    """Fit ONE per-component basis on a general dataset (robust even when the target has few
    samples/labels), freeze it, then project every target dataset's activations onto that basis
    and measure zero-shot accuracy of the PC, top-text-approx and top-image-approx reconstructions
    (both modalities), against the real-model baseline.

    Centering: each target component is centered by *its own* mean, projected onto the frozen basis,
    and the mean is added back before the (mean-restored) embedding is scored -- so a 2-row target
    still gets a full reconstruction from the general basis.

    Args mirror run_explanations. `tower` in {"image","text"}. Returns a list of row dicts
    {dataset, model, tower, components, fit_dataset, model_acc, pc_acc, image_acc, text_acc}.
    Needs, per dataset: activations (attn/mlp[_text]) and -- for scoring -- the image tower's
    {ds}_classifier_{model}.npy or the text tower's {probe_name}_embeddings/_labels.
    """
    import os
    from utils.scripts.algorithms_text_explanations import svd_data_approx

    tag = "_text" if tower == "text" else ""

    def load(ds, kind):
        return np.load(os.path.join(input_dir, f"{ds}_{kind}{tag}_{model}_seed_{seed}.npy")).astype(np.float32)

    # --- frozen candidate pools (concept labels), fit-side ---
    text_features = np.load(os.path.join(input_dir, f"{textset}_{model}.npy")).astype(np.float32)
    with open(os.path.join(text_dir, f"{textset}.txt")) as f:
        lines = [l.replace("\n", "") for l in f.readlines()]
    img_pool = None
    if image_set not in ("self", None, ""):
        for nm in (sorted({p.split(f"_embeddings_{model}_seed_{seed}.npy")[0]
                           for p in os.listdir(input_dir) if p.endswith(f"_embeddings_{model}_seed_{seed}.npy")
                           and "_text_" not in p}) if image_set == "all" else
                   [s.strip() for s in image_set.split(",")]):
            ep = os.path.join(input_dir, f"{nm}_embeddings_{model}_seed_{seed}.npy")
            if os.path.exists(ep):
                e = np.load(ep).astype(np.float32)
                img_pool = e if img_pool is None else np.concatenate([img_pool, e], 0)

    # --- FIT: one (vh, coeff_text, coeff_image) per decomposed component on the general set ---
    fit_attn, fit_mlp = load(fit_dataset, "attn"), load(fit_dataset, "mlp")
    L, H = fit_attn.shape[1], fit_attn.shape[2]
    do_attn, do_mlp = components in ("attn", "all"), components in ("mlp", "all")
    sa, sm = max(0, L - num_of_last_layers), max(0, fit_mlp.shape[1] - num_of_last_layers)

    def fit_unit(data):
        _, info = svd_data_approx(data, text_features, lines, 0, 0, text_per_pc, device, iters=max_text)
        vh = np.asarray(info["vh"], np.float32)                          # [r, d]
        coeff_text = np.asarray(info["project_matrix"], np.float32)      # [r, r] in PC space
        src = img_pool if img_pool is not None else data                # image span basis source
        Cc = src - src.mean(0, keepdims=True)
        coords = (Cc / (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-8)) @ vh.T
        pm = coords[[int(np.argmax(coords[:, pc])) for pc in range(coords.shape[1])]]
        coeff_image = (pm.T @ np.linalg.pinv(pm @ pm.T) @ pm).astype(np.float32)
        return vh, coeff_text, coeff_image

    ops = []  # (kind, l, h, vh, coeff_text, coeff_image)
    if do_attn:
        for l in range(sa, L):
            for h in range(H):
                ops.append(("attn", l, h) + fit_unit(fit_attn[:, l, h]))
    if do_mlp:
        for l in range(sm, fit_mlp.shape[1]):
            ops.append(("mlp", l, None) + fit_unit(fit_mlp[:, l]))
    print(f"[transfer] fit {tower} basis on '{fit_dataset}' ({model}): {len(ops)} components frozen.")

    # --- scoring per tower ---
    if tower == "image":
        def scorer(ds):
            cf = os.path.join(input_dir, f"{ds}_classifier_{model}.npy")
            if not os.path.exists(cf):
                return None
            classifier = np.load(cf).astype(np.float32)                 # [d, C]
            labels = np.load(os.path.join(input_dir, f"{ds}_labels_{model}_seed_{seed}.npy"))
            return lambda emb: float(((emb @ classifier).argmax(1) == labels).mean() * 100.0)
    else:
        pe = os.path.join(input_dir, f"{probe_name}_embeddings_{model}_seed_{seed}.npy")
        pl = os.path.join(input_dir, f"{probe_name}_labels_{model}_seed_{seed}.npy")
        probe_emb = np.load(pe).astype(np.float32) if os.path.exists(pe) else None
        probe_lab = np.load(pl) if os.path.exists(pl) else None

        def scorer(ds):
            if probe_emb is None:
                return None
            tl = np.load(os.path.join(input_dir, f"{ds}_labels_text_{model}_seed_{seed}.npy"))
            C = int(tl.max()) + 1

            def acc(emb):
                proto = np.stack([emb[tl == c].mean(0) for c in range(C)])
                return float(((probe_emb @ proto.T).argmax(1) == probe_lab).mean() * 100.0)
            return acc

    # --- EVAL: project each target onto the frozen basis, score every reconstruction ---
    rows = []
    for ds in eval_datasets:
        try:
            ev_attn, ev_mlp = load(ds, "attn"), load(ds, "mlp")
        except FileNotFoundError:
            continue
        score = scorer(ds)
        base = ev_mlp.sum(1) + ev_attn.sum((1, 2))                       # [N, d] real model
        emb_pc, emb_tx, emb_im = base.copy(), base.copy(), base.copy()
        for kind, l, h, vh, ct, ci in ops:
            X = ev_attn[:, l, h] if kind == "attn" else ev_mlp[:, l]     # [N, d]
            mean_e = X.mean(0, keepdims=True)
            proj = (X - mean_e) @ vh.T                                   # [N, r]  (centered)
            emb_pc += (proj @ vh + mean_e) - X                           # mean added back
            emb_tx += (proj @ ct @ vh + mean_e) - X
            emb_im += (proj @ ci @ vh + mean_e) - X
        rows.append({
            "dataset": ds, "model": model, "tower": tower, "components": components,
            "fit_dataset": fit_dataset,
            "model_acc": score(base) if score else None,
            "pc_acc": score(emb_pc) if score else None,
            "image_acc": score(emb_im) if score else None,
            "text_acc": score(emb_tx) if score else None,
        })
        print(f"[transfer] eval '{ds}': " + (", ".join(
            f"{k}={rows[-1][k]:.2f}" for k in ("model_acc", "pc_acc", "image_acc", "text_acc")
            if rows[-1][k] is not None) or "no scorer (missing classifier/probe)"))
    return rows
