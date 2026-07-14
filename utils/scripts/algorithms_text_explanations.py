
import numpy as np
import torch


@torch.no_grad()
def svd_data_approx(data, text_features, texts, layer, head, text_per_princ_comp, device, iters=None):
    print(f"\nLayer [{layer}], Head: {head}")

    """
    This function derive a text explanation of the principal components of the activation matrix
    of each (head, layer). It retrieves an approximation using the closest cosine similarity projected text
    in principal space with the given principal component.

    Args:
        data: The attention head matrix.
        text_features: The text features matrix (clip embedding).
        texts: Original text descriptions.
        layer: The current layer.
        head: The current head.
        text_per_princ_comp: Number of texts to consider for each princ_comp.
        device: The device to perform computations on.
        iters: The maximum number of PCs to use

    Returns:
        reconstruct: The reconstructed attention head matrix using the found basis.
        results: Jsonline file containing the found basis and metadata.
    """

    # Svd of attention head matrix (mean centered)
    # Center text and image data (modality gap)
    mean_values_att = np.mean(data, axis=0)
    mean_values_text = np.mean(text_features, axis=0)
    data = torch.from_numpy(data - mean_values_att).float().to(device)
    text_features = torch.from_numpy(text_features - mean_values_text).float().to(device)
   
    # Perform SVD of data matrix
    u, s, vh = torch.linalg.svd(data, full_matrices=False)
    # Total sum of singular values
    total_variance = torch.sum(s)
    # Cumulative sum of singular values
    cumulative_variance = torch.cumsum(s, dim=0)
    # Determine the rank where cumulative variance exceeds the threshold of total variance
    threshold = 0.99 # How much variance should cover the top princ_comps of the matrix 
    rank = torch.sum(cumulative_variance / total_variance < threshold).item() + 1
    if iters is not None:
        rank = min(rank, iters)
        
    vh = vh[:rank, :]
    s = s[:rank]
    u = u[:, :rank]

    # Use lower rank version of the data matrix
    data_orig = data @ vh.T
    data = u @ torch.diag_embed(s)
    # Get the projection of text embeddings into head activations matrix space
    text_features_norm = (text_features.norm(dim=-1, keepdim=True))   
    text_features = text_features / text_features_norm @ vh.T
    
    # Return the closest text_features in singular space of data matrix of top iters princ_comp
    simil_matrix = text_features.T  # [rank, M] strongest contribution of each text feature to the princ_comps
    # NB: no torch.squeeze -- with rank == 1 (e.g. a 2-row class-name text set) it would collapse the
    # [rank, M] sort to 1-D and break the 2-D slice below.
    indexes_max = torch.argsort(simil_matrix, dim=-1, descending=True)[:rank, :text_per_princ_comp]
    indexes_min = torch.argsort(simil_matrix, dim=-1)[:rank, :text_per_princ_comp]

    # Total strength princ_comps
    s = s**2
    tot_str = torch.sum(s)

    # Reconstruct
    results = []
    # Use only top indexes for reconstructing matrix
    indexes_reconstruct = indexes_max[:, 0]

    cosine_similarity = simil_matrix[:, indexes_max[:, 0]].T
    for i, (idx_max, idx_min) in enumerate(zip(indexes_max, indexes_min)):
        text_pos = []
        text_neg = []
        for k in range(text_per_princ_comp):
            idx = idx_max[k].item()
            text_pos.append({f"text_max_{k}":texts[idx], f"corr_max_{k}": simil_matrix[i, idx].item()})
        for k in range(text_per_princ_comp):
            idx = idx_min[k].item()
            text_neg.append({f"text_min_{k}":texts[idx], f"corr_min_{k}": simil_matrix[i, idx].item()})
        
        # Write them in order of the highest correlation (either positive or negative)
        corr_sign = torch.abs(simil_matrix[i, idx_max[0].item()]) > torch.abs(simil_matrix[i, idx_min[0].item()])
        text = text_pos + text_neg if corr_sign else text_neg + text_pos
        # text = sorted(text, key=lambda x: np.abs(list(x.values())[1]), reverse=True)
        # indexes_reconstruct[i] = idx_min[0] if "min" in list(text[0].keys())[0] else idx_max[0]
        results.append({"text": text, "eigen_v_emb": vh[i].tolist(), "strength_abs": s[i].item(), \
                        "strength_rel": (100 * s[i] / tot_str).item(), "cosine_similarity": cosine_similarity[i, i].item(),
                        "correlation": (text_features[indexes_max[:, 0], :])[i, i].item()})   # Reconstruct original matrix with new basis
    reconstruct = torch.zeros_like(data)


    project_matrix = text_features[indexes_reconstruct, :]
    
    # Least Square (data - A @ project_matrix) = 0 <-> A = data @ project_matrix.T @ (project_matrix @ project_matrix.T)^-1
    coefficient = project_matrix.T @ torch.linalg.pinv(project_matrix @ project_matrix.T) @ project_matrix
    
    # Reconstruct the original matrix (equivalent but faster compared to: data_orig @ project_matrix_pinv @ project_matrix)
    # This project rows of data_original (i.e. activations embeddings) into the space span by the rows of project_matrix
    reconstruct = (data_orig @ coefficient) @ vh
    
    # Json information on the procedure
    json_object = {
        "mean_values_att": mean_values_att.tolist(),
        "mean_values_text": mean_values_text.tolist(),
        "project_matrix": coefficient.tolist(),
        "vh": vh.tolist(),
        "s": s.tolist(),
        "embeddings_sort": results
    }


    return reconstruct.detach().cpu().numpy() + mean_values_att, json_object

