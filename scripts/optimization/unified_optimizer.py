import torch
import pandas as pd
import numpy as np
import copy
import os
from datetime import datetime
from scripts.optimization.bayesian import BayesianOptimizer
from scripts.optimization.random_optimizer import RandomOptimizer
from scripts.optimization.optuna import OptunaOptimizer
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator

class UnifiedHyperparameterOptimizer:
    """
    Unified interface for different hyperparameter optimization methods.
    Supports Bayesian optimization, random search, Optuna, and Population-Based Training.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir="optimization_results"):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = log_dir
        self.results = []
        self.best_auc = 0.0  # Track best AUC across all trials
        self.best_accuracy = 0.0  # Track best accuracy across all trials
        
        # Create main log directory - use local path if log_dir contains /content
        if "/content" in self.log_dir:
            self.log_dir = "optimization_results"
        # Also check if the path starts with /content
        if self.log_dir.startswith("/content"):
            self.log_dir = "optimization_results"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize different optimizers
        self.bayesian_optimizer = BayesianOptimizer(model_type, model_name, device, f"{log_dir}/bayesian")
        self.random_optimizer = RandomOptimizer(model_type, model_name, device, f"{log_dir}/random")
        self.optuna_optimizer = OptunaOptimizer(model_type, model_name, device, f"{log_dir}/optuna")
    
    def sample_initial_hyperparameters(self, mode, population_size):
        """
        Sample initial hyperparameters for the population.
        
        Args:
            mode: Training mode
            population_size: Size of the population
            
        Returns:
            List of hyperparameter dictionaries
        """
        np.random.seed(42)
        
        population = []
        for _ in range(population_size):
            # Sample learning rate (log-uniform)
            lr = np.exp(np.random.uniform(np.log(1e-5), np.log(1e-2)))
            
            # Sample batch size (discrete)
            batch_size = np.random.choice([16, 32, 64, 128])
            
            # Sample internal layer size (discrete)
            internal_layer_size = np.random.choice([64, 128, 256, 512])
            
            # Sample optimizer
            optimizer_name = np.random.choice(["adam", "adamw", "sgd"])
            
            # Sample weight decay
            weight_decay = np.exp(np.random.uniform(np.log(1e-6), np.log(1e-3)))
            
            if mode in ["supcon", "infonce"]:
                temperature = np.exp(np.random.uniform(np.log(0.01), np.log(1.0)))
                params = {
                    'lr': lr,
                    'batch_size': batch_size,
                    'internal_layer_size': internal_layer_size,
                    'optimizer': optimizer_name,
                    'weight_decay': weight_decay,
                    'temperature': temperature
                }
            else:
                margin = np.random.uniform(0.1, 2.0)
                params = {
                    'lr': lr,
                    'batch_size': batch_size,
                    'internal_layer_size': internal_layer_size,
                    'optimizer': optimizer_name,
                    'weight_decay': weight_decay,
                    'margin': margin
                }
            
            population.append(params)
        
        return population
    
    def mutate_hyperparameters(self, params, mode, mutation_rate=0.2):
        """
        Mutate hyperparameters for evolution.
        
        Args:
            params: Current hyperparameters
            mode: Training mode
            mutation_rate: Probability of mutation for each parameter
            
        Returns:
            New hyperparameter dictionary
        """
        new_params = copy.deepcopy(params)
        
        # Mutate learning rate
        if np.random.random() < mutation_rate:
            new_params['lr'] *= np.exp(np.random.normal(0, 0.5))
            new_params['lr'] = np.clip(new_params['lr'], 1e-5, 1e-2)
        
        # Mutate batch size
        if np.random.random() < mutation_rate:
            new_params['batch_size'] = np.random.choice([16, 32, 64, 128])
        
        # Mutate internal layer size
        if np.random.random() < mutation_rate:
            new_params['internal_layer_size'] = np.random.choice([64, 128, 256, 512])
        
        # Mutate optimizer
        if np.random.random() < mutation_rate:
            new_params['optimizer'] = np.random.choice(["adam", "adamw", "sgd"])
        
        # Mutate weight decay
        if np.random.random() < mutation_rate:
            new_params['weight_decay'] *= np.exp(np.random.normal(0, 0.5))
            new_params['weight_decay'] = np.clip(new_params['weight_decay'], 1e-6, 1e-3)
        
        # Mutate mode-specific parameters
        if mode in ["supcon", "infonce"]:
            if np.random.random() < mutation_rate:
                new_params['temperature'] *= np.exp(np.random.normal(0, 0.5))
                new_params['temperature'] = np.clip(new_params['temperature'], 0.01, 1.0)
        else:
            if np.random.random() < mutation_rate:
                new_params['margin'] *= np.exp(np.random.normal(0, 0.3))
                new_params['margin'] = np.clip(new_params['margin'], 0.1, 2.0)
        
        return new_params
    
    def optimize(self, method, training_filepath, test_filepath,
                mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None, validate_filepath=None, curriculum=None, **kwargs):
        """
        Run hyperparameter optimization using the specified method.
        """
        # Filter kwargs for each optimizer
        if method == "bayesian":
            allowed = ["n_calls", "n_random_starts", "epochs"]
            filtered = {k: kwargs[k] for k in allowed if k in kwargs}
            return self._run_bayesian_optimization(
                training_filepath, test_filepath,
                mode, loss_type, medium_filepath, easy_filepath, validate_filepath=validate_filepath, curriculum=curriculum, **filtered
            )
        elif method == "random":
            allowed = ["n_trials", "epochs"]
            filtered = {k: kwargs[k] for k in allowed if k in kwargs}
            return self._run_random_optimization(
                training_filepath, test_filepath,
                mode, loss_type, medium_filepath, easy_filepath, validate_filepath=validate_filepath, curriculum=curriculum, **filtered
            )
        elif method == "optuna":
            allowed = ["n_trials", "sampler", "pruner", "study_name", "epochs"]
            filtered = {k: kwargs[k] for k in allowed if k in kwargs}
            return self._run_optuna_optimization(
                training_filepath, test_filepath,
                mode, loss_type, medium_filepath, easy_filepath, validate_filepath=validate_filepath, curriculum=curriculum, **filtered
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _run_bayesian_optimization(self, training_filepath, test_filepath,
                                 mode, loss_type, medium_filepath, easy_filepath, validate_filepath=None, curriculum=None, **kwargs):
        """Run Bayesian optimization."""
        return self.bayesian_optimizer.optimize(
            training_filepath, test_filepath,
            mode, loss_type, medium_filepath, easy_filepath, validate_filepath=validate_filepath, curriculum=curriculum, **kwargs
        )
    
    def _run_random_optimization(self, training_filepath, test_filepath,
                               mode, loss_type, medium_filepath, easy_filepath, validate_filepath=None, curriculum=None, **kwargs):
        """Run random search optimization."""
        return self.random_optimizer.optimize(
            training_filepath, test_filepath,
            mode, loss_type, medium_filepath, easy_filepath, validate_filepath=validate_filepath, curriculum=curriculum, **kwargs
        )
    
    def _run_optuna_optimization(self, training_filepath, test_filepath,
                               mode, loss_type, medium_filepath, easy_filepath, validate_filepath=None, curriculum=None, **kwargs):
        """Run Optuna optimization."""
        return self.optuna_optimizer.optimize(
            training_filepath, test_filepath,
            mode, loss_type, medium_filepath, easy_filepath, validate_filepath=validate_filepath, curriculum=curriculum, **kwargs
        )
    
    def compare_methods(self, training_filepath, test_filepath,
                       mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None, curriculum=None, **kwargs):
        """
        Compare different optimization methods on the same problem.
        """
        print(f"Comparing optimization methods for {self.model_type} model")
        print(f"Mode: {mode}, Loss: {loss_type}")
        
        methods = ["random", "bayesian", "optuna"]
        results = {}
        for method in methods:
            print(f"\n{'='*50}")
            print(f"Running {method.upper()} optimization")
            print(f"{'='*50}")
            try:
                # Filter kwargs for each optimizer
                if method == "bayesian":
                    allowed = ["n_calls", "n_random_starts", "epochs"]
                elif method == "random":
                    allowed = ["n_trials", "epochs"]
                elif method == "optuna":
                    allowed = ["n_trials", "sampler", "pruner", "study_name", "epochs"]
                filtered = {k: kwargs[k] for k in allowed if k in kwargs}
                method_results = self.optimize(
                    method, training_filepath, test_filepath,
                    mode, loss_type, medium_filepath, easy_filepath, curriculum=curriculum, **filtered
                )
                results[method] = method_results
                if method_results:
                    best_auc = max(r.get('test_auc', 0) for r in method_results)
                    best_accuracy = max(r.get('test_accuracy', 0) for r in method_results)
                    print(f"{method.upper()} - Best AUC: {best_auc:.4f}, Best Accuracy: {best_accuracy:.4f}")
            except Exception as e:
                print(f"Error in {method} optimization: {e}")
                results[method] = []
        self._save_comparison_results(results, mode, loss_type)
        return results
    
    def _save_comparison_results(self, results, mode, loss_type):
        """Save comparison results to CSV."""
        all_results = []
        for method, method_results in results.items():
            for result in method_results:
                result['optimization_method'] = method
                all_results.append(result)
        
        if all_results:
            df = pd.DataFrame(all_results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/comparison_results_{mode}_{loss_type}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Comparison results saved to {filename}")
    
    def get_recommended_settings(self, mode, loss_type, dataset_size=None):
        """
        Get recommended hyperparameter settings based on the model type and mode.
        
        Args:
            mode: Training mode
            loss_type: Loss function type
            dataset_size: Size of the dataset (optional)
            
        Returns:
            Dictionary with recommended settings
        """
        # Base recommendations
        recommendations = {
            'clip': {
                'pair': {'lr': 1e-4, 'batch_size': 32, 'internal_layer_size': 128},
                'triplet': {'lr': 1e-4, 'batch_size': 64, 'internal_layer_size': 256},
                'supcon': {'lr': 1e-4, 'batch_size': 32, 'internal_layer_size': 128, 'temperature': 0.1},
                'infonce': {'lr': 1e-4, 'batch_size': 32, 'internal_layer_size': 128, 'temperature': 0.1}
            },
            'flava': {
                'pair': {'lr': 5e-5, 'batch_size': 16, 'internal_layer_size': 256},
                'triplet': {'lr': 5e-5, 'batch_size': 32, 'internal_layer_size': 512},
                'supcon': {'lr': 5e-5, 'batch_size': 16, 'internal_layer_size': 256, 'temperature': 0.07},
                'infonce': {'lr': 5e-5, 'batch_size': 16, 'internal_layer_size': 256, 'temperature': 0.07}
            },
            'siglip': {
                'pair': {'lr': 1e-4, 'batch_size': 32, 'internal_layer_size': 128},
                'triplet': {'lr': 1e-4, 'batch_size': 64, 'internal_layer_size': 256},
                'supcon': {'lr': 1e-4, 'batch_size': 32, 'internal_layer_size': 128, 'temperature': 0.1},
                'infonce': {'lr': 1e-4, 'batch_size': 32, 'internal_layer_size': 128, 'temperature': 0.1}
            }
        }
        
        # Get recommendations for the specific model type
        if self.model_type in recommendations and mode in recommendations[self.model_type]:
            base_recs = recommendations[self.model_type][mode].copy()
            
            # Adjust based on dataset size
            if dataset_size:
                if dataset_size < 1000:
                    base_recs['batch_size'] = min(base_recs['batch_size'], 16)
                    base_recs['lr'] *= 2  # Higher learning rate for small datasets
                elif dataset_size > 10000:
                    base_recs['batch_size'] = min(base_recs['batch_size'] * 2, 128)
                    base_recs['lr'] *= 0.5  # Lower learning rate for large datasets
            
            # Add common parameters
            base_recs.update({
                'optimizer': 'adamw',
                'weight_decay': 1e-4
            })
            
            return base_recs
        else:
            # Fallback recommendations
            return {
                'lr': 1e-4,
                'batch_size': 32,
                'internal_layer_size': 128,
                'optimizer': 'adamw',
                'weight_decay': 1e-4
            } 