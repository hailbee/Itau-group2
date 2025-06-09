import torch
import torch.nn.functional as F
import pandas as pd
from utils.embeddings import EmbeddingExtractor
from utils.embeddings import batched_embedding

# Assumes: EmbeddingExtractor and batched_embedding are already defined


# test_model(trained_model_path, reference_data_file_path, test_file_path):
#       reference_names = utilHelperFunc(reference_data_file_path)
#       test_names, test_labels = helperfunc(test_file_path)
def test_pair_model(trained_model, reference_filepath, test_filepath, batch_size=32):
    reference_names = pd.read_csv(reference_filepath)['normalized_company'].tolist()
    test_names = pd.read_csv(test_filepath)['company'].tolist()
    test_labels = pd.read_csv(test_filepath)['label'].tolist()

    extractor = EmbeddingExtractor(trained_model)
    legit_embeddings = batched_embedding(extractor, reference_names, batch_size)
    test_embeddings = batched_embedding(extractor, test_names, batch_size)

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

def test_triplet_model(trained_model, anchor_filepath, positive_filepath, negative_filepath, batch_size=32):
    # Load triplet names
    anchor_names = pd.read_csv(anchor_filepath)['company'].tolist()
    positive_names = pd.read_csv(positive_filepath)['company'].tolist()
    negative_names = pd.read_csv(negative_filepath)['company'].tolist()

    extractor = EmbeddingExtractor(trained_model)

    # Get embeddings
    anchor_emb = batched_embedding(extractor, anchor_names, batch_size)
    positive_emb = batched_embedding(extractor, positive_names, batch_size)
    negative_emb = batched_embedding(extractor, negative_names, batch_size)

    # Compute distances
    pos_dist = F.pairwise_distance(anchor_emb, positive_emb, p=2)
    neg_dist = F.pairwise_distance(anchor_emb, negative_emb, p=2)

    # Compute margin ranking accuracy (how often positive is closer than negative)
    correct = (pos_dist < neg_dist).sum().item()
    total = len(pos_dist)
    accuracy = correct / total

    # Build dataframe of results
    results = pd.DataFrame({
        "anchor": anchor_names,
        "positive": positive_names,
        "negative": negative_names,
        "pos_dist": pos_dist.tolist(),
        "neg_dist": neg_dist.tolist(),
        "is_correct": (pos_dist < neg_dist).tolist()
    })

    print(f"Triplet accuracy: {accuracy:.4f}")
    return results