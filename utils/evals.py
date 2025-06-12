import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score

def find_best_threshold_precision(y_true, y_scores, thresholds):
    """
    Finds the best threshold that yields the highest precision.

    Args:
        y_true (list or np.ndarray): Ground truth binary labels.
        y_scores (list or np.ndarray): Predicted similarity scores.
        thresholds (list or np.ndarray): Thresholds to test.

    Returns:
        float: Best precision
        float: Threshold that gives best precision
    """
    best_prec = 0
    best_prec_threshold = 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec > best_prec:
            best_prec = prec
            best_prec_threshold = t

    print(f"Best Precision: {best_prec:.4f} at Threshold: {best_prec_threshold:.3f}")
    return best_prec, best_prec_threshold


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
        results_df (pd.DataFrame): DataFrame containing 'label' and 'max_similarity'.
    Returns:
        float: Best threshold maximizing TPR - FPR.
    """
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
    Finds the best threshold that yields the highest accuracy.

    Args:
        results_df (pd.DataFrame): Must contain 'label' and 'max_similarity'.
    Returns:
        float: Best accuracy
        float: Threshold that gives best accuracy
    """
    best_acc = 0
    best_acc_threshold = 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_acc_threshold = t

    print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
    return best_acc, best_acc_threshold

def find_best_threshold_precision(y_true, y_scores, thresholds):
    """
    Finds the best threshold that yields the highest precision.

    Args:
        y_true (list or np.ndarray): Ground truth binary labels.
        y_scores (list or np.ndarray): Predicted similarity scores.
        thresholds (list or np.ndarray): Thresholds to test.

    Returns:
        float: Best precision
        float: Threshold that gives best precision
    """
    best_prec = 0
    best_prec_threshold = 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        if prec > best_prec:
            best_prec = prec
            best_prec_threshold = t

    print(f"Best Precision: {best_prec:.4f} at Threshold: {best_prec_threshold:.3f}")
    return best_prec, best_prec_threshold
