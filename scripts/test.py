import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from utils.utils import EmbeddingExtractor

# Assumes: EmbeddingExtractor and batched_embedding are already defined


# test_model(trained_model_path, reference_data_file_path, test_file_path):
#       legit_names = utilHelperFunc(reference_data_file_path)
#       test_names, test_labels = helperfunc(test_file_path)
def test_model(trained_model, legit_names, test_names, test_labels, batch_size=32):
    extractor = EmbeddingExtractor(trained_model)
    legit_embeddings = batched_embedding(extractor, legit_names, batch_size)
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
        matched_name = legit_names[idx]
        label = test_labels[i]

        results.append({
            "name": test_name,
            "label": label,
            "max_similarity": max_sim.item(),
            "matched_name": matched_name
        })

    return pd.DataFrame(results)