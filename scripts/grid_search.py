import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_curve, accuracy_score

from utils.utils import ContrastiveLoss
from scripts.train import train
from scripts.test import test_model
from scripts.eval import evaluate_model

def grid_search(dataset, device, reference_filepath, test_filepath, lrs, batch_sizes, margins, internal_layer_sizes, model_class):
    results = []
    best_acc = 0
    best_config = {}

    for batch_size in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for internal_layer_size in internal_layer_sizes:
            for lr in lrs:
                for margin in margins:
                    model = model_class(embedding_dim=512, projection_dim=internal_layer_size).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                    criterion = ContrastiveLoss(margin=margin)

                    best_model_state = None
                    best_loss = float("inf")

                    print(f"\n--- Training config: lr={lr}, bs={batch_size}, margin={margin}, size={internal_layer_size} ---")
                    for epoch in range(5):
                        epoch_loss = train(model, dataloader, criterion, optimizer, device)
                        if epoch_loss < best_loss:
                            best_loss = epoch_loss
                            best_model_state = model.state_dict()

                    # Save model
                    model_path = f"model_lr{lr}_bs{batch_size}_m{margin}_ils{internal_layer_size}.pth"
                    if best_model_state is not None:
                        torch.save(best_model_state, model_path)

                    # Load best weights back
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    model.eval()

                    # Evaluate
                    print(f"--- Evaluating config: lr={lr}, bs={batch_size}, margin={margin}, size={internal_layer_size} ---")
                    results_df_eval = test_model(model, reference_filepath, test_filepath, batch_size=batch_size)
                    evaluation_metrics = evaluate_model(results_df_eval)

                    y_true = results_df_eval['label']
                    y_scores = results_df_eval['max_similarity']
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

                    current_best_acc = 0
                    current_best_thresh = 0
                    for t in thresholds:
                        y_pred = (y_scores > t).astype(int)
                        acc = accuracy_score(y_true, y_pred)
                        if acc > current_best_acc:
                            current_best_acc = acc
                            current_best_thresh = t

                    if current_best_acc > best_acc:
                        best_acc = current_best_acc
                        best_config = {
                            "lr": lr,
                            "batch_size": batch_size,
                            "margin": margin,
                            "internal_layer_size": internal_layer_size,
                            "best_loss": best_loss,
                            "best_accuracy": current_best_acc,
                            "threshold": current_best_thresh
                        }

                    results.append({
                        "timestamp": datetime.now(),
                        "lr": lr,
                        "batch_size": batch_size,
                        "margin": margin,
                        "internal_layer_size": internal_layer_size,
                        "epochs": 5,
                        "best_train_loss": best_loss,
                        "test_auc": evaluation_metrics["roc_auc"],
                        "test_youden_threshold": evaluation_metrics["youden_threshold"],
                        "test_best_accuracy": current_best_acc,
                        "test_accuracy_threshold": current_best_thresh
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("experiment_results_with_accuracy.csv", index=False)

    print("\nOverall Best config based on max test accuracy:")
    print(best_config)
