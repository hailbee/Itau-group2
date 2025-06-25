import torch
import pandas as pd
from datetime import datetime
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet
from model_utils.models.supcon import SiameseCLIPSupCon
from model_utils.models.infonce import SiameseCLIPInfoNCE

class GridSearcher:
    """
    Unified grid search interface for hyperparameter optimization.
    """
    def __init__(self, model_class, device, log_dir="grid_search_results"):
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir

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

    def search(self, reference_filepath, test_reference_filepath, test_filepath,
              lrs, batch_sizes, margins, internal_layer_sizes,
              mode="pair", loss_type="cosine", warmup_filepath=None,
              epochs=5, warmup_epochs=5, cirriculum = None, temperature=0.07):
        """
        Perform grid search over hyperparameters.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            lrs: List of learning rates to try
            batch_sizes: List of batch sizes to try
            margins: List of margins to try (used for margin-based losses)
            internal_layer_sizes: List of internal layer sizes to try
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            warmup_filepath: Optional path to warmup data
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            temperature: Temperature parameter for SupCon/InfoNCE loss
        """
        results = []
        best_loss = float("inf")
        best_acc = 0
        best_config = {}

        # Load data
        dataframe = pd.read_pickle(reference_filepath)
        if warmup_filepath:
            warmup_dataframe = pd.read_pickle(warmup_filepath)

        # Get loss class
        loss_class = self.get_loss_class(mode, loss_type)

        for batch_size in batch_sizes:
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, batch_size, mode)
            warmup_loader = None
            if warmup_filepath:
                warmup_loader = self.create_dataloader(warmup_dataframe, batch_size, mode)

            for internal_layer_size in internal_layer_sizes:
                for lr in lrs:
                    for margin in margins:
                        # Create model and optimizer
                        model = self.model_class(
                            embedding_dim=512,
                            projection_dim=internal_layer_size
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
                            log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        )
                        evaluator = Evaluator(model, batch_size=batch_size)

                        print(f"\n--- Training config: lr={lr}, bs={batch_size}, "
                              f"{'temperature' if mode in ['supcon', 'infonce'] else 'margin'}={temperature if mode in ['supcon', 'infonce'] else margin}, "
                              f"size={internal_layer_size}, loss={loss_type} ---")

                        # Train model
                        model_loss = trainer.train(
                            dataloader=dataloader,
                            test_reference_filepath=test_reference_filepath,
                            test_filepath=test_filepath,
                            mode=mode,
                            epochs=epochs,
                            warmup_loader=warmup_loader,
                            warmup_epochs=warmup_epochs,
                            cirriculum=cirriculum
                        )

                        # Evaluate model
                        results_df, metrics = evaluator.evaluate(
                            test_reference_filepath,
                            test_filepath
                        )

                        # Track best results
                        if model_loss < best_loss:
                            best_loss = model_loss
                            best_model_state = model.state_dict()
                            best_model_path = (
                                f"{self.log_dir}/model_lr{lr}_bs{batch_size}_"
                                f"{'temp' if mode in ['supcon', 'infonce'] else 'm'}{temperature if mode in ['supcon', 'infonce'] else margin}_"
                                f"ils{internal_layer_size}_{mode}_{loss_type}.pth"
                            )
                            torch.save(best_model_state, best_model_path)

                        if metrics['accuracy'] > best_acc:
                            best_acc = metrics['accuracy']
                            best_config = {
                                "lr": lr,
                                "batch_size": batch_size,
                                "temperature" if mode in ["supcon", "infonce"] else "margin": temperature if mode in ["supcon", "infonce"] else margin,
                                "internal_layer_size": internal_layer_size,
                                "best_loss": best_loss,
                                "best_accuracy": metrics['accuracy'],
                                "threshold": metrics['threshold'],
                                "loss_type": loss_type
                            }

                        # Log results
                        results.append({
                            "timestamp": datetime.now(),
                            "lr": lr,
                            "batch_size": batch_size,
                            "temperature" if mode in ["supcon", "infonce"] else "margin": temperature if mode in ["supcon", "infonce"] else margin,
                            "internal_layer_size": internal_layer_size,
                            "epochs": epochs,
                            "best_train_loss": best_loss,
                            "test_auc": metrics['roc_curve'][1].mean(),  # Mean TPR
                            "test_youden_threshold": metrics['threshold'],
                            "test_best_accuracy": metrics['accuracy'],
                            "test_accuracy_threshold": metrics['threshold'],
                            "loss_type": loss_type
                        })

        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.log_dir}/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         index=False)

        print("\nOverall Best config based on max test accuracy:")
        print(best_config)

        return best_config, results_df

    def create_dataloader(self, dataframe, batch_size, mode):
        """Create appropriate dataloader based on mode"""
        if mode == "pair":
            from utils.data import TextPairDataset
            dataset = TextPairDataset(dataframe)
        elif mode == "triplet":
            from utils.data import TripletDataset
            dataset = TripletDataset(dataframe)
        elif mode == "supcon":
            from utils.data import SupConDataset
            dataset = SupConDataset(dataframe)
        elif mode == "infonce":
            from utils.data import InfoNCEDataset
            dataset = InfoNCEDataset(dataframe)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4) 