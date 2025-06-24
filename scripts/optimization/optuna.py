import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet
from model_utils.models.supcon import SiameseCLIPSupCon
from model_utils.models.infonce import SiameseCLIPInfoNCE

class OptunaOptimizer:
    """
    Optuna-based hyperparameter optimization with multiple samplers and pruning.
    Provides advanced optimization algorithms and visualization capabilities.
    """
    
    def __init__(self, model_class, device, log_dir="optuna_optimization_results"):
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir
        self.results = []
        self.best_auc = 0.0  # Track best AUC across all trials
        self.best_accuracy = 0.0  # Track best accuracy across all trials
        
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
    
    def objective(self, trial, reference_filepath, test_reference_filepath, test_filepath,
                 mode, loss_type, warmup_filepath=None, epochs=5, warmup_epochs=5):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            warmup_filepath: Optional warmup data path
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            
        Returns:
            float: Objective value (accuracy)
        """
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        internal_layer_size = trial.suggest_categorical("internal_layer_size", [64, 128, 256, 512])
        
        if mode in ["supcon", "infonce"]:
            temperature = trial.suggest_float("temperature", 0.01, 1.0, log=True)
            margin_or_temp = temperature
        else:
            margin = trial.suggest_float("margin", 0.1, 2.0)
            margin_or_temp = margin
        
        # Optional: suggest optimizer
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        
        # Optional: suggest weight decay
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        try:
            # Load data
            dataframe = pd.read_pickle(reference_filepath)
            warmup_dataframe = None
            if warmup_filepath:
                warmup_dataframe = pd.read_pickle(warmup_filepath)
            
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, batch_size, mode)
            warmup_loader = None
            if warmup_dataframe is not None:
                warmup_loader = self.create_dataloader(warmup_dataframe, batch_size, mode)
            
            # Create model
            model = self.model_class(
                embedding_dim=512,
                projection_dim=internal_layer_size
            ).to(self.device)
            
            # Create optimizer
            if optimizer_name == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:  # sgd
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Get loss class and create criterion
            loss_class = self.get_loss_class(mode, loss_type)
            if mode in ["supcon", "infonce"]:
                criterion = loss_class(temperature=temperature)
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
            
            # Train model
            model_loss = trainer.train(
                dataloader=dataloader,
                test_reference_filepath=test_reference_filepath,
                test_filepath=test_filepath,
                mode=mode,
                epochs=epochs,
                warmup_loader=warmup_loader,
                warmup_epochs=warmup_epochs
            )
            
            # Evaluate model
            results_df, metrics = evaluator.evaluate(
                test_reference_filepath,
                test_filepath
            )
            
            # Log results
            result = {
                "trial_number": trial.number,
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": batch_size,
                "internal_layer_size": internal_layer_size,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "epochs": epochs,
                "train_loss": model_loss,
                "test_accuracy": metrics['accuracy'],
                "test_auc": metrics['roc_auc'],
                "threshold": metrics['threshold'],
                "loss_type": loss_type,
                **{k: v for k, v in locals().items() if k in ['temperature', 'margin'] and v is not None}
            }
            self.results.append(result)
            
            # Track best AUC
            current_auc = metrics['roc_auc']
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                print(f"Trial {trial.number}: New best AUC = {self.best_auc:.4f}")
            else:
                print(f"Trial {trial.number}: AUC = {current_auc:.4f} (Best = {self.best_auc:.4f})")
            
            # Track best accuracy
            current_accuracy = metrics['accuracy']
            if current_accuracy > self.best_accuracy:
                self.best_accuracy = current_accuracy
                print(f"Trial {trial.number}: New best accuracy = {self.best_accuracy:.4f}")
            else:
                print(f"Trial {trial.number}: Accuracy = {current_accuracy:.4f} (Best = {self.best_accuracy:.4f})")
            
            # Report intermediate value for pruning
            trial.report(metrics['accuracy'], epochs)
            
            return metrics['accuracy']
            
        except Exception as e:
            print(f"Error in trial {trial.number}: {e}")
            return 0.0
    
    def optimize(self, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None,
                epochs=5, warmup_epochs=5, n_trials=50, sampler="tpe", 
                pruner="median", study_name=None):
        """
        Perform Optuna-based hyperparameter optimization.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            warmup_filepath: Optional path to warmup data
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            n_trials: Number of optimization trials
            sampler: Sampler type ("tpe", "random", "cmaes")
            pruner: Pruner type ("median", "hyperband", None)
            study_name: Name for the study (for storage)
        """
        print(f"Starting Optuna optimization for {mode} mode with {loss_type} loss")
        print(f"Sampler: {sampler}, Pruner: {pruner}, Trials: {n_trials}")
        
        # Create study name
        if study_name is None:
            study_name = f"{mode}_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Set up sampler
        if sampler == "tpe":
            sampler_obj = TPESampler(seed=42)
        elif sampler == "random":
            sampler_obj = RandomSampler(seed=42)
        elif sampler == "cmaes":
            sampler_obj = CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Set up pruner
        if pruner == "median":
            pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        elif pruner == "hyperband":
            pruner_obj = HyperbandPruner()
        elif pruner is None:
            pruner_obj = None
        else:
            raise ValueError(f"Unknown pruner: {pruner}")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler_obj,
            pruner=pruner_obj,
            study_name=study_name,
            storage=None  # Use in-memory storage instead of SQLite
        )
        
        # Create objective function with fixed parameters
        def objective_wrapper(trial):
            return self.objective(
                trial, reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, epochs, warmup_epochs
            )
        
        # Run optimization
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # Get best results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_accuracy = best_trial.value
        
        print(f"\nOptimization completed!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.log_dir}/optuna_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         index=False)
        
        # Print best AUC
        if not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")
        
        # Save study
        study_path = f"{self.log_dir}/optuna_study_{study_name}.pkl"
        with open(study_path, "wb") as f:
            import pickle
            pickle.dump(study, f)
        
        # Create best configuration
        best_config = {
            **best_params,
            "best_accuracy": best_accuracy,
            "mode": mode,
            "loss_type": loss_type,
            "optimization_method": "optuna",
            "sampler": sampler,
            "pruner": pruner
        }
        
        return best_config, results_df, study
    
    def visualize_results(self, study, save_dir=None):
        """
        Create visualization plots for the optimization results.
        
        Args:
            study: Optuna study object
            save_dir: Directory to save plots (defaults to log_dir)
        """
        if save_dir is None:
            save_dir = self.log_dir
        
        try:
            import matplotlib.pyplot as plt
            
            # Optimization history
            fig, ax = plt.subplots(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax)
            plt.title("Optimization History")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/optimization_history.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter importance
            fig, ax = plt.subplots(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study, ax=ax)
            plt.title("Parameter Importance")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/parameter_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Parameter relationships
            fig, ax = plt.subplots(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=ax)
            plt.title("Parameter Relationships")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/parameter_relationships.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Visualization plots saved to {save_dir}")
            
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Error creating visualizations: {e}") 