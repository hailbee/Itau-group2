import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet
from model_utils.models.supcon import SiameseCLIPSupCon
from model_utils.models.infonce import SiameseCLIPInfoNCE

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning using Gaussian Processes.
    More efficient than grid search, especially for continuous hyperparameters.
    """
    
    def __init__(self, model_class, device, log_dir="bayesian_optimization_results"):
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir
        self.results = []
        
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
    
    def objective_function(self, params, reference_filepath, test_reference_filepath, 
                          test_filepath, mode, loss_type, warmup_filepath=None, 
                          epochs=5, warmup_epochs=5):
        """
        Objective function for Bayesian optimization.
        Returns negative accuracy (since we want to maximize accuracy).
        """
        # Unpack parameters
        lr, batch_size, margin_or_temp, internal_layer_size = params
        
        try:
            # Load data
            dataframe = pd.read_pickle(reference_filepath)
            warmup_dataframe = None
            if warmup_filepath:
                warmup_dataframe = pd.read_pickle(warmup_filepath)
            
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, int(batch_size), mode)
            warmup_loader = None
            if warmup_dataframe is not None:
                warmup_loader = self.create_dataloader(warmup_dataframe, int(batch_size), mode)
            
            # Create model and optimizer
            model = self.model_class(
                embedding_dim=512,
                projection_dim=int(internal_layer_size)
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Get loss class and create criterion
            loss_class = self.get_loss_class(mode, loss_type)
            if mode in ["supcon", "infonce"]:
                criterion = loss_class(temperature=margin_or_temp)
            elif mode == "triplet" and loss_type == "hybrid":
                criterion = loss_class(margin=margin_or_temp, alpha=0.5)
            else:
                criterion = loss_class(margin=margin_or_temp)
            
            # Create trainer and evaluator
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            evaluator = Evaluator(model, batch_size=int(batch_size))
            
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
                "timestamp": datetime.now(),
                "lr": lr,
                "batch_size": int(batch_size),
                "margin_or_temp": margin_or_temp,
                "internal_layer_size": int(internal_layer_size),
                "epochs": epochs,
                "train_loss": model_loss,
                "test_accuracy": metrics['accuracy'],
                "test_auc": metrics['roc_curve'][1].mean(),
                "threshold": metrics['threshold'],
                "loss_type": loss_type
            }
            self.results.append(result)
            
            print(f"Trial - lr={lr:.2e}, bs={int(batch_size)}, "
                  f"{'temp' if mode in ['supcon', 'infonce'] else 'margin'}={margin_or_temp:.3f}, "
                  f"size={int(internal_layer_size)}, acc={metrics['accuracy']:.4f}")
            
            # Return negative accuracy (we want to maximize accuracy)
            return -metrics['accuracy']
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return -0.0  # Return worst possible score
    
    def optimize(self, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None,
                epochs=5, warmup_epochs=5, n_calls=50, n_random_starts=10):
        """
        Perform Bayesian optimization over hyperparameters.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            warmup_filepath: Optional path to warmup data
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            n_calls: Number of optimization iterations
            n_random_starts: Number of random initial points
        """
        print(f"Starting Bayesian optimization for {mode} mode with {loss_type} loss")
        print(f"Will perform {n_calls} trials with {n_random_starts} random starts")
        
        # Define search space
        if mode in ["supcon", "infonce"]:
            # For SupCon/InfoNCE, use temperature instead of margin
            search_space = [
                Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
                Integer(16, 128, name='batch_size'),
                Real(0.01, 1.0, prior='log-uniform', name='temperature'),
                Integer(64, 512, name='internal_layer_size')
            ]
            param_names = ['lr', 'batch_size', 'temperature', 'internal_layer_size']
        else:
            # For pair/triplet, use margin
            search_space = [
                Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
                Integer(16, 128, name='batch_size'),
                Real(0.1, 2.0, prior='uniform', name='margin'),
                Integer(64, 512, name='internal_layer_size')
            ]
            param_names = ['lr', 'batch_size', 'margin', 'internal_layer_size']
        
        # Create objective function with fixed parameters
        @use_named_args(search_space)
        def objective(**params):
            param_values = [params[name] for name in param_names]
            return self.objective_function(
                param_values, reference_filepath, test_reference_filepath,
                test_filepath, mode, loss_type, warmup_filepath, epochs, warmup_epochs
            )
        
        # Run optimization
        result = gp_minimize(
            func=objective,
            dimensions=search_space,
            n_calls=n_calls,
            n_random_starts=n_random_starts,
            noise=0.1,  # Add noise to handle training variability
            verbose=True
        )
        
        # Get best parameters
        best_params = dict(zip(param_names, result.x))
        best_accuracy = -result.fun  # Convert back from negative
        
        print(f"\nOptimization completed!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.log_dir}/bayesian_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         index=False)
        
        # Save best configuration
        best_config = {
            **best_params,
            "best_accuracy": best_accuracy,
            "mode": mode,
            "loss_type": loss_type,
            "optimization_method": "bayesian"
        }
        
        return best_config, results_df