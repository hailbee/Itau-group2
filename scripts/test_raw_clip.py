import torch
import torch.nn.functional as F
import pandas as pd
from utils.embeddings import get_clip_embeddings

def test_raw_clip(reference_filepath, test_filepath):
    reference_names = pd.read_csv(reference_filepath)['normalized_company'].tolist()
    test_df = pd.read_csv(test_filepath)
    test_names = test_df['company'].tolist()
    test_labels = test_df['label'].tolist()

    #get clip embeddings for each
    legit_embeddings = get_clip_embeddings(reference_names)  # shape: [N_legit, D]
    test_embeddings = get_clip_embeddings(test_names)

    similarity_matrix = F.cosine_similarity(
        test_embeddings.unsqueeze(1),
        legit_embeddings.unsqueeze(0),
        dim=2
    )

    results = []
    for i, test_name in enumerate(test_names):
        sims = similarity_matrix[i]
        max_sim, idx = torch.max(sims, dim=0)
        matched_name = reference_names[idx]
        label = test_labels[i]

        results.append({
            "name": test_name,
            "label": label,
            "max_similarity": max_sim.item(),
            "matched_name": matched_name
        })

    return pd.DataFrame(results)