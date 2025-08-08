import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score
import numpy as np

def plot_roc_curve(results_df):
    """
    Plots the ROC curve given a DataFrame with 'label' and 'max_similarity' columns.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'label' and 'max_similarity' columns.
    Returns:
        roc_auc (float): Computed AUC value
        fpr, tpr, thresholds (np.ndarray): ROC curve components
    """
    y_true = results_df['label']
    y_scores = results_df['max_similarity']

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return roc_auc, fpr, tpr, thresholds

def find_best_threshold_youden(fpr, tpr, thresholds):
    """
    Finds the best threshold based on Youden's J statistic (TPR - FPR).

    Args:
        fpr (np.ndarray): False positive rates from ROC curve
        tpr (np.ndarray): True positive rates from ROC curve  
        thresholds (np.ndarray): Thresholds from ROC curve
    Returns:
        float: Best threshold maximizing TPR - FPR.
    """
    # Vectorized computation of Youden's J statistic
    youden_index = tpr - fpr
    best_idx = youden_index.argmax()
    best_threshold = thresholds[best_idx]
    print(f"Best threshold (Youden): {best_threshold:.3f}")
    return best_threshold

def plot_confusion_matrix(y_true, y_scores, threshold):
    """
    Plots a confusion matrix using a specified threshold to binarize predictions.

    Args:
        results_df (pd.DataFrame): Must contain 'label' and 'max_similarity'.
        threshold (float): Threshold for classifying scores into binary labels.
    """
    y_pred = (y_scores > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spoof', 'Spoof'])

    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix at Threshold = {threshold:.3f}')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def find_best_threshold_accuracy(y_true, y_scores, thresholds):
    """
    Finds the best threshold that yields the highest accuracy using vectorized operations.
    
    Args:
        y_true (np.ndarray): True labels
        y_scores (np.ndarray): Predicted scores
        thresholds (np.ndarray): Thresholds to evaluate
    Returns:
        float: Best accuracy
        float: Threshold that gives best accuracy
    """
    # Convert to numpy arrays for vectorized operations
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    thresholds = np.asarray(thresholds)
    
    # Filter out inf and nan values from thresholds
    valid_mask = np.isfinite(thresholds)
    if not np.any(valid_mask):
        # If no valid thresholds, return default values
        return 0.0, 0.0
    
    valid_thresholds = thresholds[valid_mask]
    
    # Vectorized computation: for each threshold, compute predictions for all samples at once
    # Shape: (n_thresholds, n_samples)
    predictions = (y_scores[:, None] > valid_thresholds[None, :]).astype(int)
    
    # Vectorized accuracy computation: compare predictions with true labels
    # Shape: (n_thresholds,)
    accuracies = np.mean(predictions == y_true[:, None], axis=0)
    
    # Find the best accuracy and corresponding threshold
    best_idx = np.argmax(accuracies)
    best_acc = accuracies[best_idx]
    best_acc_threshold = valid_thresholds[best_idx]
    
    print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
    return best_acc, best_acc_threshold
