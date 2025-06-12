import argparse
import ast
import torch
from torch.utils.data import DataLoader
import pandas as pd
from datetime import datetime
from sklearn.metrics import roc_curve, accuracy_score

from utils.loss import CosineLoss, EuclideanLoss, CosineTripletLoss, EuclideanTripletLoss, HybridTripletLoss
from utils.data import TextPairDataset, TripletDataset
from scripts.train import train_pair, train_triplet, train_triplet_warmup
from scripts.test import test_model
from scripts.eval import evaluate_model
from models.models import SiameseCLIPModelPairs, SiameseCLIPTriplet

def grid_search(reference_filepath, test_reference_set_filepath, test_filepath, lrs, batch_sizes, margins, internal_layer_sizes, mode="pair", loss_type="cosine", warmup_filepath=None):
    print("grid search began")
    results = []
    best_loss = float("inf")
    best_acc = 0
    best_config = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataframe = pd.read_pickle(reference_filepath)
    if warmup_filepath:
        warmup_dataframe = pd.read_pickle(warmup_filepath)

    if mode == "pair":
        dataset = TextPairDataset(dataframe)
        train_func = train_pair
        model_class = SiameseCLIPModelPairs
        if loss_type == "cosine":
            loss_class = CosineLoss
        elif loss_type == "euclidean":
            loss_class = EuclideanLoss
        else:
            raise ValueError("Unsupported loss_type for pair mode.")

    elif mode == "triplet":
        dataset = TripletDataset(dataframe)
        train_func = train_triplet
        if warmup_filepath:
            train_func = train_triplet_warmup
            warmup_dataset = TripletDataset(warmup_dataframe)
        model_class = SiameseCLIPTriplet
        if loss_type == "cosine":
            loss_class = CosineTripletLoss
        elif loss_type == "euclidean":
            loss_class = EuclideanTripletLoss
        elif loss_type == "hybrid":
            loss_class = HybridTripletLoss
        else:
            raise ValueError("Unsupported loss_type for triplet mode.")

    else:
        raise ValueError("Unsupported mode. Use 'pair' or 'triplet'.")

    for batch_size in batch_sizes:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if warmup_filepath:
            warmup_loader = DataLoader(warmup_dataset, batch_size=batch_size, shuffle=True)
        for internal_layer_size in internal_layer_sizes:
            for lr in lrs:
                for margin in margins:
                    model = model_class(embedding_dim=512, projection_dim=internal_layer_size).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    if loss_type == "hybrid" and mode == "triplet":
                        criterion = HybridTripletLoss(margin=margin, alpha=0.5)
                    else:
                        criterion = loss_class(margin=margin)

                    best_model_state = None

                    print(f"\n--- Training config: lr={lr}, bs={batch_size}, margin={margin}, size={internal_layer_size}, loss={loss_type} ---")
                    if warmup_filepath:
                        model_loss = train_func(model, warmup_loader, dataloader, criterion, optimizer, device)
                    else:
                        model_loss = train_func(model, dataloader, criterion, optimizer, device)
                    if model_loss < best_loss:
                        best_loss = model_loss
                        best_model_state = model.state_dict()
                        best_model_path = f"model_lr{lr}_bs{batch_size}_m{margin}_ils{internal_layer_size}_{mode}_{loss_type}.pth"

                    model.eval()

                    print(f"--- Evaluating config: lr={lr}, bs={batch_size}, margin={margin}, size={internal_layer_size}, loss={loss_type} ---")
                    results_df_eval = test_model(model, test_reference_set_filepath, test_filepath, batch_size=batch_size)
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
                            "threshold": current_best_thresh,
                            "loss_type": loss_type
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
                        "test_accuracy_threshold": current_best_thresh,
                        "loss_type": loss_type
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv("experiment_results_with_accuracy.csv", index=False)

    if best_model_state is not None:
        torch.save(best_model_state, best_model_path)

    print("\nOverall Best config based on max test accuracy:")
    print(best_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_filepath", type=str)
    parser.add_argument("--test_reference_set_filepath", type=str)
    parser.add_argument("--test_filepath", type=str)
    parser.add_argument("--lrs", type=str)
    parser.add_argument("--batch_size", type=str)
    parser.add_argument("--margins", type=str)
    parser.add_argument("--internal_layer_sizes", type=str)
    parser.add_argument("--mode", type=str, default="pair")
    parser.add_argument("--loss_type", type=str, default="cosine")
    parser.add_argument("--warmup_filepath", type=str, default=None)

    args = parser.parse_args()

    lrs = ast.literal_eval(args.lrs)
    batch_sizes = ast.literal_eval(args.batch_size)
    margins = ast.literal_eval(args.margins)
    internal_layer_sizes = ast.literal_eval(args.internal_layer_sizes)

    grid_search(
        reference_filepath=args.reference_filepath,
        test_reference_set_filepath=args.test_reference_set_filepath,
        test_filepath=args.test_filepath,
        lrs=lrs,
        batch_sizes=batch_sizes,
        margins=margins,
        internal_layer_sizes=internal_layer_sizes,
        mode=args.mode,
        loss_type=args.loss_type,
        warmup_filepath=args.warmup_filepath
    )