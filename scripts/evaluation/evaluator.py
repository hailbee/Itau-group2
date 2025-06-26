import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from utils.evals import find_best_threshold_youden
from utils.embeddings import EmbeddingExtractor, batched_embedding

class Evaluator:
    """
    Unified evaluation interface for model testing and metrics computation.
    """
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.extractor = EmbeddingExtractor(model)

    def compute_similarities(self, reference_names, test_names):
        """Compute similarity matrix between reference and test names"""
        legit_embeddings = batched_embedding(self.extractor, reference_names, self.batch_size)
        test_embeddings = batched_embedding(self.extractor, test_names, self.batch_size)

        return F.cosine_similarity(
            test_embeddings.unsqueeze(1),
            legit_embeddings.unsqueeze(0),
            dim=2
        )

    def test_model(self, reference_filepath, test_filepath):
        """
        Test model on reference and test data.
        
        Args:
            reference_filepath: Path to reference data CSV
            test_filepath: Path to test data CSV
            
        Returns:
            pd.DataFrame: Results with predictions and metrics
        """
        # Load data
        reference_names = pd.read_csv(reference_filepath)['normalized_company'].tolist()
        test_data = pd.read_csv(test_filepath)
        test_names = test_data['company'].tolist()
        test_labels = test_data['label'].tolist()

        # Compute similarities
        similarity_matrix = self.compute_similarities(reference_names, test_names)

        # Process results
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

    def compute_metrics(self, results_df):
        """
        Compute evaluation metrics from results.
        
        Args:
            results_df: DataFrame with test results
            
        Returns:
            dict: Dictionary of metrics
        """
        y_true = results_df['label']
        y_scores = results_df['max_similarity']
        
        # Compute ROC curve and find optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds)
        
        # Compute predictions using optimal threshold
        y_pred = (y_scores > youden_thresh).astype(int)
        
        # Compute metrics
        roc_auc = roc_auc_score(y_true, y_scores)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'threshold': youden_thresh,
            'roc_curve': (fpr, tpr, thresholds),
            'roc_auc': roc_auc
        }
        
        return metrics

    def evaluate(self, reference_filepath, test_filepath):
        """
        Complete evaluation pipeline.
        
        Args:
            reference_filepath: Path to reference data
            test_filepath: Path to test data
            
        Returns:
            tuple: (results_df, metrics)
        """
        results_df = self.test_model(reference_filepath, test_filepath)
        metrics = self.compute_metrics(results_df)
        return results_df, metrics 