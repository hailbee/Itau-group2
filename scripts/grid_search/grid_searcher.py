import torch
import pandas as pd
from datetime import datetime
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.learning import (
    SiameseModelPairs, 
    SiameseModelTriplet,
    SiameseModelSupCon,
    SiameseModelInfoNCE
)

class GridSearcher:
    """
    Unified grid search interface for hyperparameter optimization.
    """
    def __init__(self, model_class, device, log_dir="grid_search_results", backbone=None):
        import os
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir
        self.backbone = backbone
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

    def get_loss_class(self, mode, loss_type):
        """Get appropriate loss class based on mode and type"""
        if mode == "pair":
            if loss_type == "cosine":
                from model_utils.loss.pair_losses import CosineLoss
                return CosineLoss
            elif loss_type == "euclidean":
                from model_utils.loss.pair_losses import EuclideanLoss
                return EuclideanLoss
        elif mode == "triplet":
            if loss_type == "cosine":
                from model_utils.loss.triplet_losses import CosineTripletLoss
                return CosineTripletLoss
            elif loss_type == "euclidean":
                from model_utils.loss.triplet_losses import EuclideanTripletLoss
                return EuclideanTripletLoss
            elif loss_type == "hybrid":
                from model_utils.loss.triplet_losses import HybridTripletLoss
                return HybridTripletLoss
        elif mode == "supcon":
            if loss_type == "supcon":
                from model_utils.loss.supcon_loss import SupConLoss
                return SupConLoss
            elif loss_type == "infonce":
                from model_utils.loss.infonce_loss import InfoNCELoss
                return InfoNCELoss
        elif mode == "infonce":
            from model_utils.loss.infonce_loss import InfoNCELoss
            return InfoNCELoss
        raise ValueError(f"Unsupported mode/loss_type combination: {mode}/{loss_type}")

    def search(self, training_filepath, test_reference_filepath, test_filepath,
              lrs, batch_sizes, margins, internal_layer_sizes,
              mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None,
              epochs=5, temperature=0.07):
        """
        Perform grid search over hyperparameters.
        
        Args:
            training_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            lrs: List of learning rates to try
            batch_sizes: List of batch sizes to try
            margins: List of margins to try (used for margin-based losses)
            internal_layer_sizes: List of internal layer sizes to try
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            epochs: Number of training epochs
            temperature: Temperature parameter for SupCon/InfoNCE loss
        """
        results = []
        best_loss = float("inf")
        best_acc = 0
        best_auc = 0.0
        best_config = {}

        # Load data
        dataframe = pd.read_parquet(training_filepath)
        if medium_filepath and easy_filepath:
            medium_dataframe = pd.read_parquet(medium_filepath)
            easy_dataframe = pd.read_parquet(easy_filepath)

        # Get loss class
        loss_class = self.get_loss_class(mode, loss_type)

        for batch_size in batch_sizes:
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, batch_size, mode)
            medium_loader = None
            easy_loader = None
            if medium_filepath and easy_filepath:
                medium_loader = self.create_dataloader(medium_dataframe, batch_size, mode)
                medium_easy = self.create_dataloader(easy_dataframe, batch_size, mode)

            for internal_layer_size in internal_layer_sizes:
                for lr in lrs:
                    for margin in margins:
                        # Ensure embedding_dim and projection_dim are ints
                        embedding_dim = self.backbone.embedding_dim
                        if isinstance(embedding_dim, (tuple, list)):
                            embedding_dim = embedding_dim[0]
                        projection_dim = internal_layer_size
                        if isinstance(projection_dim, (tuple, list)):
                            projection_dim = projection_dim[0]
                        # Ensure all numeric parameters are proper types
                        embedding_dim = int(embedding_dim)
                        projection_dim = int(projection_dim)
                        lr = float(lr)
                        margin = float(margin)
                        batch_size = int(batch_size)
                        internal_layer_size = int(internal_layer_size)
                        
                        # Create model and optimizer
                        model = self.model_class(
                            embedding_dim,
                            projection_dim
                        ).to(self.device)
                        
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                        # Create loss function based on mode
                        if mode in ["supcon", "infonce"]:
                            criterion = loss_class(temperature=temperature)  # Use temperature param
                        elif mode == "triplet" and loss_type == "hybrid":
                            criterion = loss_class(margin=margin, alpha=0.5)
                        else:
                            criterion = loss_class(margin=margin)

                        # Create trainer and evaluator
                        trainer = Trainer(
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            device=self.device,
                            log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            model_type=mode
                        )
                        evaluator = Evaluator(model, batch_size=batch_size, model_type=mode)

                        print(f"\n--- Training config: lr={lr}, bs={batch_size}, "
                              f"{'temperature' if mode in ['supcon', 'infonce'] else 'margin'}={temperature if mode in ['supcon', 'infonce'] else margin}, "
                              f"size={internal_layer_size}, loss={loss_type} ---")

                        # Train model
                        train_kwargs = dict(
                            dataloader=dataloader,
                            test_reference_filepath=test_reference_filepath,
                            test_filepath=test_filepath,
                            mode=mode,
                            epochs=epochs,
                            medium_loader=medium_loader,
                            easy_loader=easy_loader
                        )
                        if hasattr(self, 'curriculum') and self.curriculum is not None:
                            train_kwargs['curriculum'] = self.curriculum
                        best_metrics = trainer.train(**train_kwargs)

                        # Track best results
                        if best_metrics.get('loss', float('inf')) < best_loss:
                            best_loss = best_metrics.get('loss', float('inf'))
                            best_model_state = model.state_dict()
                            best_model_path = (
                                f"{self.log_dir}/model_lr{lr}_bs{batch_size}_"
                                f"{'temp' if mode in ['supcon', 'infonce'] else 'm'}{temperature if mode in ['supcon', 'infonce'] else margin}_"
                                f"ils{internal_layer_size}_{mode}_{loss_type}.pth"
                            )
                            torch.save(best_model_state, best_model_path)

                        if best_metrics['accuracy'] > best_acc:
                            best_acc = best_metrics['accuracy']
                            best_config = {
                                "lr": lr,
                                "batch_size": batch_size,
                                "temperature" if mode in ["supcon", "infonce"] else "margin": temperature if mode in ["supcon", "infonce"] else margin,
                                "internal_layer_size": internal_layer_size,
                                "best_loss": best_loss,
                                "best_accuracy": best_metrics['accuracy'],
                                "threshold": best_metrics.get('threshold', 0.5),
                                "loss_type": loss_type
                            }

                        # Track best AUC
                        current_auc = best_metrics['roc_auc']
                        if current_auc > best_auc:
                            best_auc = current_auc
                            print(f"New best AUC: {best_auc:.4f}")
                        
                        print(f"Config: lr={lr}, bs={batch_size}, {'temp' if mode in ['supcon', 'infonce'] else 'margin'}={temperature if mode in ['supcon', 'infonce'] else margin}, size={internal_layer_size}")
                        print(f"Accuracy: {best_metrics['accuracy']:.4f} (Best: {best_acc:.4f}), AUC: {current_auc:.4f} (Best: {best_auc:.4f})")

                        # Log results
                        results.append({
                            "timestamp": datetime.now(),
                            "lr": lr,
                            "batch_size": batch_size,
                            "temperature" if mode in ["supcon", "infonce"] else "margin": temperature if mode in ["supcon", "infonce"] else margin,
                            "internal_layer_size": internal_layer_size,
                            "epochs": epochs,
                            "best_train_loss": best_loss,
                            "test_auc": best_metrics['roc_auc'],
                            "test_youden_threshold": best_metrics.get('threshold', 0.5),
                            "test_best_accuracy": best_metrics['accuracy'],
                            "test_accuracy_threshold": best_metrics.get('threshold', 0.5),
                            "loss_type": loss_type
                        })

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.log_dir}/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         index=False)

        print("\nOverall Best config based on max test accuracy:")
        print(best_config)

        # Print best AUC
        if not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")

        return best_config, results_df

    def create_dataloader(self, dataframe, batch_size, mode):
        """Create appropriate dataloader based on mode"""
        if mode == "pair":
            from utils.data import TextPairDataset
            dataset = TextPairDataset(dataframe)
            from torch.utils.data import DataLoader
            num_workers = 0
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif mode == "triplet":
            from utils.data import TripletDataset
            dataset = TripletDataset(dataframe)
            from torch.utils.data import DataLoader
            num_workers = 4
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        elif mode == "supcon":
            from utils.data import SupConDataset, supcon_collate_fn
            dataset = SupConDataset(dataframe)
            from torch.utils.data import DataLoader
            num_workers = 4
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=supcon_collate_fn)
        elif mode == "infonce":
            from utils.data import InfoNCEDataset, infonce_collate_fn
            dataset = InfoNCEDataset(dataframe)
            from torch.utils.data import DataLoader
            num_workers = 4
            return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=infonce_collate_fn)
        else:
            raise ValueError(f"Unknown mode: {mode}")