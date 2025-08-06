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
        
    def optimize(self, training_filepath, test_filepath,
                mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None,
                epochs=5, n_trials=50, validate_filepath=None, curriculum=None):
        """
        Run random search optimization.
        
        Args:
            training_filepath: Path to training data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            epochs: Number of training epochs per trial
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
                params,
                training_filepath=training_filepath,
                test_filepath=test_filepath,
                mode=mode,
                loss_type=loss_type,
                medium_filepath=medium_filepath,
                epochs=epochs,
                easy_filepath=easy_filepath,
                validate_filepath=validate_filepath,
                curriculum=curriculum
            )
            print(f"\nTrial {i+1} completed.")
        
        # Save results
        self._save_results()

        # After all trials, evaluate the best model on the test set
        import torch, json, os
        print("[DEBUG] Evaluating best model on test set after all trials...")
        best_model_path = os.path.join(self.log_dir, 'best_model.pt')
        best_hparams_path = os.path.join(self.log_dir, 'best_hparams.json')
        if os.path.exists(best_model_path) and os.path.exists(best_hparams_path):
            with open(best_hparams_path, 'r') as f:
                best_params = json.load(f)
            model = self.create_siamese_model(mode, int(best_params.get('internal_layer_size', 128))).to(self.device)
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            evaluator = Evaluator(model, batch_size=int(best_params.get('batch_size', 32)), model_type=mode)
            model.eval()
            _, test_metrics = evaluator.evaluate(test_filepath)
            print("[DEBUG] Final test set evaluation:", test_metrics)
            return test_metrics
        else:
            print("[DEBUG] No best model found for final test set evaluation.")
            return self.results
    
    def _save_results(self):
        """Save optimization results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/random_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}") 