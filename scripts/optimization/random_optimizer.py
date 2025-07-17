import torch
import pandas as pd
import numpy as np
import os
from datetime import datetime
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from .base_optimizer import BaseOptimizer

class RandomOptimizer(BaseOptimizer):
    """
    Random search for hyperparameter optimization.
    Often more effective than grid search, especially in high-dimensional spaces.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir="random_optimization_results"):
        super().__init__(model_type, model_name, device, log_dir)
        
    def optimize(self, training_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None,
                epochs=5, warmup_epochs=5, n_trials=50, validate_filepath=None):
        """
        Run random search optimization.
        
        Args:
            training_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            warmup_filepath: Optional warmup data path
            epochs: Number of training epochs per trial
            warmup_epochs: Number of warmup epochs
            n_trials: Number of random trials
        """
        print(f"Starting random search optimization for {self.model_type} model")
        print(f"Mode: {mode}, Loss: {loss_type}")
        print(f"Will run {n_trials} trials")
        
        # Sample hyperparameters
        trials = self.sample_hyperparameters(mode, n_trials)
        
        # Evaluate each trial
        for i, params in enumerate(trials):
            print(f"\n{'='*50}")
            print(f"Starting Trial {i+1}")
            print(f"{'='*50}")
            
            result = self.evaluate_trial(
                params, training_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, epochs, warmup_epochs, validate_filepath
            )
            print(f"\nTrial {i+1} completed - AUC: {result.get('test_auc', 0):.4f}, Accuracy: {result.get('test_accuracy', 0):.4f}")
        
        # Save results
        self._save_results()
        
        print(f"\n{'='*60}")
        print(f"Random search completed!")
        print(f"Best AUC: {self.best_auc:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"{'='*60}")
        
        return self.results
    
    def _save_results(self):
        """Save optimization results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/random_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}") 