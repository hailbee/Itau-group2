import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_curve, precision_score, recall_score, accuracy_score, roc_auc_score
from utils.evals import find_best_threshold_youden, plot_roc_curve, plot_confusion_matrix, find_best_threshold_accuracy
from utils.embeddings import EmbeddingExtractor, SupConEmbeddingExtractor, batched_embedding
import numpy as np
from sklearn.metrics import auc

class Evaluator:
    """
    Unified evaluation interface for model testing and metrics computation (pairwise only).
    """
    def __init__(self, model, batch_size=32, model_type=None):
        self.model = model
        self.batch_size = batch_size
        self.model_type = model_type
        # Only use embedding extractor
        if model_type in ['supcon', 'infonce']:
            print("USING SUPCON EMBEDDING EXTRACTOR")
            self.extractor = SupConEmbeddingExtractor(model)
        else:
            print("USING STANDARD EMBEDDING EXTRACTOR")
            self.extractor = EmbeddingExtractor(model)

    def compute_metrics(self, results_df, plot=False):
        """
        Compute evaluation metrics from results.
        Optionally plot ROC curve and confusion matrix.
        Args:
            results_df: DataFrame with test results
            plot (bool): If True, plot ROC curve and confusion matrices. If False, do not plot anything.
        Returns:
            dict: Dictionary of metrics
        """
        y_true = results_df['label']
        y_scores = results_df['max_similarity']
        if plot:
            roc_auc, fpr, tpr, thresholds = plot_roc_curve(results_df)
        else:
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

        # Use the ROC curve thresholds for both calculations to avoid redundant computation
        youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds)
        best_acc, best_acc_threshold = find_best_threshold_accuracy(y_true, y_scores, thresholds)
        
        y_pred = (y_scores > youden_thresh).astype(int)
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'threshold': youden_thresh,
            'roc_curve': (fpr, tpr, thresholds),
            'roc_auc': roc_auc,
            'best_accuracy': best_acc,
            'best_accuracy_threshold': best_acc_threshold
        }
        if plot:
            plot_confusion_matrix(y_true, y_scores, youden_thresh)
            print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
            plot_confusion_matrix(y_true, y_scores, best_acc_threshold)
        return metrics

    def evaluate(self, test_filepath, plot=False):
        """
        Evaluate model on a file of (fraudulent_name, real_name, label) pairs.
        Args:
            test_filepath: Path to test data (CSV or PARQUET with fraudulent_name, real_name, label)
            plot (bool): Whether to plot ROC/confusion matrix
        Returns:
            tuple: (results_df, metrics)
        """
        return self.test_pairs(test_filepath, plot=plot)

    def test_pairs(self, test_filepath, plot=False):
        import pandas as pd
        
        if test_filepath.endswith('.csv'):
            df = pd.read_csv(test_filepath)
        else:
            df = pd.read_parquet(test_filepath)
        
        fraud_names = df['fraudulent_name'].astype(str).tolist()
        real_names = df['real_name'].astype(str).tolist()
        labels = df['label'].astype(float).tolist()
        
        fraud_embs = batched_embedding(self.extractor, fraud_names, self.batch_size)
        real_embs = batched_embedding(self.extractor, real_names, self.batch_size)
        
        similarities = F.cosine_similarity(fraud_embs, real_embs, dim=1).detach().cpu().numpy()
        results_df = pd.DataFrame({
            'fraudulent_name': fraud_names,
            'real_name': real_names,
            'label': labels,
            'max_similarity': similarities
        })
        
        metrics = self.compute_metrics(results_df, plot=plot)
        return results_df, metrics 