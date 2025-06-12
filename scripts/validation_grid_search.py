import argparse
import ast
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score

from utils.loss import CosineLoss, EuclideanLoss, CosineTripletLoss, EuclideanTripletLoss, HybridTripletLoss
from utils.data import TextPairDataset, TripletDataset
from scripts.train_and_validate import train_and_val_pair, train_and_val_triplet
from models.models import SiameseCLIPModelPairs, SiameseCLIPTriplet

def validation_grid_search(reference_filepath, test_reference_set_filepath, test_filepath, lrs, batch_sizes, margins, internal_layer_sizes, mode="pair", loss_type="cosine"):
    print("grid search began")
    results = []
    best_loss = float("inf")
    best_acc = 0
    best_config = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataframe = pd.read_pickle(reference_filepath)

    if mode == "pair":
        dataset = TextPairDataset(dataframe)
        train_func = train_and_val_pair
        model_class = SiameseCLIPModelPairs
        if loss_type == "cosine":
            loss_class = CosineLoss
        elif loss_type == "euclidean":
            loss_class = EuclideanLoss
        else:
            raise ValueError("Unsupported loss_type for pair mode.")

    elif mode == "triplet":
        dataset = TripletDataset(dataframe)
        train_func = train_and_val_triplet
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
        for internal_layer_size in internal_layer_sizes:
            for lr in lrs:
                for margin in margins:
                    model = model_class(embedding_dim=512, projection_dim=internal_layer_size).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    if loss_type == "hybrid" and mode == "triplet":
                        criterion = HybridTripletLoss(margin=margin, alpha=0.5)
                    else:
                        criterion = loss_class(margin=margin)

                    print(f"\n--- Training config: lr={lr}, bs={batch_size}, margin={margin}, size={internal_layer_size}, loss={loss_type} ---")
                    model_loss = train_func(
                        model,
                        dataloader,
                        criterion,
                        optimizer,
                        device,
                        test_reference_set_filepath,
                        test_filepath
                    )

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

    args = parser.parse_args()

    lrs = ast.literal_eval(args.lrs)
    batch_sizes = ast.literal_eval(args.batch_size)
    margins = ast.literal_eval(args.margins)
    internal_layer_sizes = ast.literal_eval(args.internal_layer_sizes)

    validation_grid_search(
        reference_filepath=args.reference_filepath,
        test_reference_set_filepath=args.test_reference_set_filepath,
        test_filepath=args.test_filepath,
        lrs=lrs,
        batch_sizes=batch_sizes,
        margins=margins,
        internal_layer_sizes=internal_layer_sizes,
        mode=args.mode,
        loss_type=args.loss_type
    )