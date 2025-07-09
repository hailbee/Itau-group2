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
from .base_optimizer import BaseOptimizer

class OptunaOptimizer(BaseOptimizer):
    """
    Optuna-based hyperparameter optimization with multiple samplers and pruning.
    Provides advanced optimization algorithms and visualization capabilities.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir="optuna_optimization_results"):
        super().__init__(model_type, model_name, device, log_dir)
        
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
        
        # Create parameter dictionary
        params = {
            'lr': lr,
            'batch_size': batch_size,
            'internal_layer_size': internal_layer_size,
            'optimizer': optimizer_name,
            'weight_decay': weight_decay
        }
        
        if mode in ["supcon", "infonce"]:
            params['temperature'] = temperature
        else:
            params['margin'] = margin
        
        try:
            # Use the base class evaluate_trial method
            result = self.evaluate_trial(
                params, reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, epochs, warmup_epochs
            )
            
            # Add trial number to result
            result["trial_number"] = trial.number
            
            return result.get('test_accuracy', 0.0)
            
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            return 0.0
    
    def optimize(self, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None,
                epochs=5, warmup_epochs=5, n_trials=50, sampler="tpe", 
                pruner="median", study_name=None):
        """
        Run Optuna optimization.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            warmup_filepath: Optional warmup data path
            epochs: Number of training epochs per trial
            warmup_epochs: Number of warmup epochs
            n_trials: Number of trials
            sampler: Sampler type ("tpe", "random", "cmaes")
            pruner: Pruner type ("median", "hyperband")
            study_name: Name for the study
        """
        print(f"Starting Optuna optimization for {self.model_type} model")
        print(f"Mode: {mode}, Loss: {loss_type}")
        print(f"Sampler: {sampler}, Pruner: {pruner}")
        print(f"Will run {n_trials} trials")
        
        # Create study name if not provided
        if study_name is None:
            study_name = f"{self.model_type}_{mode}_{loss_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Choose sampler
        if sampler == "tpe":
            sampler_obj = TPESampler(seed=42)
        elif sampler == "random":
            sampler_obj = RandomSampler(seed=42)
        elif sampler == "cmaes":
            sampler_obj = CmaEsSampler(seed=42)
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        
        # Choose pruner
        if pruner == "median":
            pruner_obj = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner == "hyperband":
            pruner_obj = HyperbandPruner(min_resource=1, max_resource=epochs)
        else:
            raise ValueError(f"Unknown pruner: {pruner}")
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler_obj,
            pruner=pruner_obj,
            study_name=study_name
        )
        
        # Define objective wrapper
        def objective_wrapper(trial):
            return self.objective(
                trial, reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, epochs, warmup_epochs
            )
        
        # Run optimization
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # Save results
        self._save_results(study)
        
        print(f"Optuna optimization completed!")
        print(f"Best AUC: {self.best_auc:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best parameters: {study.best_params}")
        
        return self.results
    
    def _save_results(self, study):
        """Save optimization results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/optuna_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
            
            # Save study
            study_filename = f"{self.log_dir}/optuna_study_{timestamp}.pkl"
            study.export_artifacts(study_filename)
            print(f"Study saved to {study_filename}")