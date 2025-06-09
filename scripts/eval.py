from utils.utils import plot_roc_curve, find_best_threshold_youden, plot_confusion_matrix, find_best_threshold_accuracy

def evaluate_model(results_df):
    """
    Evaluates the model based on a DataFrame of test results.
    Includes ROC curve, Youden threshold, confusion matrix, and accuracy threshold.

    Args:
        results_df (pd.DataFrame): Must contain 'label' and 'max_similarity' columns.
    """
    # ROC & AUC
    roc_auc, fpr, tpr, thresholds = plot_roc_curve(results_df)
    print(f"AUC: {roc_auc:.4f}")

    # Youden's J threshold
    youden_thresh = find_best_threshold_youden(fpr, tpr, thresholds)

    # Plot confusion matrix at Youden threshold
    y_true = results_df['label']
    y_scores = results_df['max_similarity']
    plot_confusion_matrix(y_true, y_scores, youden_thresh)

    # Best Accuracy threshold
    best_acc, best_acc_thresh = find_best_threshold_accuracy(y_true, y_scores, thresholds)

    return {
        "roc_auc": roc_auc,
        "youden_threshold": youden_thresh,
        "best_accuracy": best_acc,
        "accuracy_threshold": best_acc_thresh
    }
