import torch
import pandas as pd
import numpy as np
from datetime import datetime
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.siamese import SiameseCLIPModelPairs, SiameseCLIPTriplet
from model_utils.models.supcon import SiameseCLIPSupCon
from model_utils.models.infonce import SiameseCLIPInfoNCE

class RandomOptimizer:
    """
    Random search for hyperparameter optimization.
    Often more effective than grid search, especially in high-dimensional spaces.
    """
    
    def __init__(self, model_class, device, log_dir="random_optimization_results"):
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir
        self.results = []
        
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
    
    def sample_hyperparameters(self, mode, n_trials):
        """
        Sample hyperparameters randomly for the specified number of trials.
        
        Args:
            mode: Training mode ("pair", "triplet", "supcon", "infonce")
            n_trials: Number of trials to sample
            
        Returns:
            List of parameter dictionaries
        """
        np.random.seed(42)  # For reproducibility
        
        trials = []
        for _ in range(n_trials):
            # Sample learning rate (log-uniform)
            lr = np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2)))
            
            # Sample batch size (discrete)
            batch_size = np.random.choice([16, 32, 64, 128])
            
            # Sample internal layer size (discrete)
            internal_layer_size = np.random.choice([64, 128, 256, 512])
            
            if mode in ["supcon", "infonce"]:
                # Sample temperature (log-uniform)
                temperature = np.exp(np.random.uniform(np.log(0.01), np.log(1.0)))
                trials.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'temperature': temperature,
                    'internal_layer_size': internal_layer_size
                })
            else:
                # Sample margin (uniform)
                margin = np.random.uniform(0.1, 2.0)
                trials.append({
                    'lr': lr,
                    'batch_size': batch_size,
                    'margin': margin,
                    'internal_layer_size': internal_layer_size
                })
        
        return trials
    
    def evaluate_trial(self, params, reference_filepath, test_reference_filepath,
                      test_filepath, mode, loss_type, warmup_filepath=None,
                      epochs=5, warmup_epochs=5):
        """
        Evaluate a single hyperparameter configuration.
        
        Returns:
            Dictionary with results
        """
        try:
            # Load data
            dataframe = pd.read_pickle(reference_filepath)
            warmup_dataframe = None
            if warmup_filepath:
                warmup_dataframe = pd.read_pickle(warmup_filepath)
            
            # Create dataloaders
            dataloader = self.create_dataloader(dataframe, params['batch_size'], mode)
            warmup_loader = None
            if warmup_dataframe is not None:
                warmup_loader = self.create_dataloader(warmup_dataframe, params['batch_size'], mode)
            
            # Create model and optimizer
            model = self.model_class(
                embedding_dim=512,
                projection_dim=params['internal_layer_size']
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            
            # Get loss class and create criterion
            loss_class = self.get_loss_class(mode, loss_type)
            if mode in ["supcon", "infonce"]:
                criterion = loss_class(temperature=params['temperature'])
            elif mode == "triplet" and loss_type == "hybrid":
                criterion = loss_class(margin=params['margin'], alpha=0.5)
            else:
                criterion = loss_class(margin=params['margin'])
            
            # Create trainer and evaluator
            trainer = Trainer(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                device=self.device,
                log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            evaluator = Evaluator(model, batch_size=params['batch_size'])
            
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
            
            # Return results
            return {
                "timestamp": datetime.now(),
                "lr": params['lr'],
                "batch_size": params['batch_size'],
                "internal_layer_size": params['internal_layer_size'],
                "epochs": epochs,
                "train_loss": model_loss,
                "test_accuracy": metrics['accuracy'],
                "test_auc": metrics['roc_curve'][1].mean(),
                "threshold": metrics['threshold'],
                "loss_type": loss_type,
                **{k: v for k, v in params.items() if k not in ['lr', 'batch_size', 'internal_layer_size']}
            }
            
        except Exception as e:
            print(f"Error in trial: {e}")
            return {
                "timestamp": datetime.now(),
                "lr": params.get('lr', 0),
                "batch_size": params.get('batch_size', 0),
                "internal_layer_size": params.get('internal_layer_size', 0),
                "epochs": epochs,
                "train_loss": float('inf'),
                "test_accuracy": 0.0,
                "test_auc": 0.0,
                "threshold": 0.0,
                "loss_type": loss_type,
                "error": str(e),
                **{k: v for k, v in params.items() if k not in ['lr', 'batch_size', 'internal_layer_size']}
            }
    
    def optimize(self, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None,
                epochs=5, warmup_epochs=5, n_trials=50):
        """
        Perform random search over hyperparameters.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            warmup_filepath: Optional path to warmup data
            epochs: Number of training epochs
            warmup_epochs: Number of warmup epochs
            n_trials: Number of random trials to perform
        """
        print(f"Starting random search for {mode} mode with {loss_type} loss")
        print(f"Will perform {n_trials} random trials")
        
        # Sample hyperparameters
        trials = self.sample_hyperparameters(mode, n_trials)
        
        best_accuracy = 0.0
        best_config = None
        
        # Evaluate each trial
        for i, params in enumerate(trials):
            print(f"\nTrial {i+1}/{n_trials}")
            print(f"Parameters: {params}")
            
            result = self.evaluate_trial(
                params, reference_filepath, test_reference_filepath,
                test_filepath, mode, loss_type, warmup_filepath, epochs, warmup_epochs
            )
            
            self.results.append(result)
            
            # Track best result
            if result['test_accuracy'] > best_accuracy:
                best_accuracy = result['test_accuracy']
                best_config = {**params, "best_accuracy": best_accuracy}
            
            print(f"Accuracy: {result['test_accuracy']:.4f}")
            print(f"Best so far: {best_accuracy:.4f}")
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f"{self.log_dir}/random_search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                         index=False)
        
        print(f"\nRandom search completed!")
        print(f"Best accuracy: {best_accuracy:.4f}")
        print(f"Best configuration: {best_config}")
        
        return best_config, results_df 