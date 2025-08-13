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

# Suppress Optuna's internal logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaOptimizer(BaseOptimizer):
    """
    Optuna-based hyperparameter optimization with multiple samplers and pruning.
    Provides advanced optimization algorithms and visualization capabilities.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir=None):
        # Hardcode log_dir to Google Drive path
        log_dir = "/content/drive/My Drive/Project_2_Business_Names/Summer 2025/code/optimizer results"
        import os
        os.makedirs(log_dir, exist_ok=True)
        super().__init__(model_type, model_name, device, log_dir)
        
    def objective(self, trial, training_filepath, test_filepath,
                 mode, loss_type, medium_filepath=None, easy_filepath=None, epochs=5, validate_filepath=None, curriculum=None):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            training_filepath: Path to training data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            epochs: Number of training epochs
        Returns:
            float: Objective value (accuracy)
        """
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        internal_layer_size = trial.suggest_categorical("internal_layer_size", [512, 768, 1024])
        

        
        params = {}
        
        if mode in ["supcon", "infonce"]:
            temperature = trial.suggest_float("temperature", 0.01, 0.1, log=True)
            params['temperature'] = temperature
        else:
            margin = trial.suggest_float("margin", 0.05, 0.7)
            params['margin'] = margin
        
        # Optional: suggest optimizer
        optimizer_name = trial.suggest_categorical("optimizer", ["adam"])
       
        # Optional: suggest weight decay
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-5, log=True)
        
        # Create parameter dictionary
        params.update({
            'lr': lr,
            'batch_size': batch_size,
            'internal_layer_size': internal_layer_size,

            'optimizer': optimizer_name,
            'weight_decay': weight_decay
        })
        try:
            print(f"\n{'='*50}")
            print(f"Starting Trial {trial.number + 1}")
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
            
            result["trial_number"] = trial.number + 1
            print(f"\nTrial {trial.number + 1} completed.")
            return result.get('test_accuracy', 0.0)
        
        except Exception as e:
            print(f"\nTrial {trial.number + 1} failed with error: {e}")
            return 0.0
    
    def optimize(self, training_filepath, test_filepath,
                mode="pair", loss_type="cosine", medium_filepath=None, easy_filepath=None,
                epochs=5, n_trials=50, sampler="tpe", 
                pruner="median", study_name=None, validate_filepath=None, curriculum=None):
        """
        Run Optuna optimization.
        
        Args:
            training_filepath: Path to training data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            epochs: Number of training epochs per trial
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
                trial, training_filepath, test_filepath,
                mode, loss_type, medium_filepath, easy_filepath, epochs, validate_filepath, curriculum
            )
        
        # Run optimization
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # Save results
        self._save_results(study)

        # After all trials, evaluate the best model on the test set
        import torch, json, os
        print("\n" + "="*60)
        print("[DEBUG] FINAL COMPARISON: Evaluating best model on test set after all Optuna trials...")
        print("="*60 + "\n")
        model_id = f"{self.model_type}_{mode}"
        # Include curriculum in filename if specified
        if curriculum:
            model_id += f"_{curriculum}"
        best_model_path = os.path.join(self.log_dir, f'best_model_{model_id}.pt')
        best_hparams_path = os.path.join(self.log_dir, f'best_hparams_{model_id}.json')
        
        if os.path.exists(best_model_path) and os.path.exists(best_hparams_path):
            with open(best_hparams_path, 'r') as f:
                best_params = json.load(f)
            # Print best_params with all float/int values rounded to 4 decimals
            rounded_best_params = {
                k: (round(v, 4) if isinstance(v, (float, int)) else v)
                for k, v in best_params.items()
            }
            print(f"[DEBUG] Initial hyperparameters of best model: {rounded_best_params}")
            model = self.create_siamese_model(mode, int(best_params.get('internal_layer_size', 128))).to(self.device)
            model.load_state_dict(torch.load(best_model_path, map_location=self.device))
            evaluator = Evaluator(model, batch_size=int(best_params.get('batch_size', 32)), model_type=mode)
            model.eval()
            _, test_metrics = evaluator.evaluate(test_filepath)
            # Print only relevant metrics (exclude 'roc_curve'), rounded to 4 decimals
            metrics_to_print = {
                k: (round(v, 4) if isinstance(v, (float, int)) else v)
                for k, v in test_metrics.items() if k != 'roc_curve'
            }

            # Pretty print key metrics
            print("\n--- FINAL TEST SET METRICS ---")
            if 'youden_j' in metrics_to_print:
                print(f"Youden's J statistic:         {metrics_to_print['youden_j']:.4f}")
            if 'top_acc_threshold' in metrics_to_print:
                print(f"Top Accuracy Threshold:       {metrics_to_print['top_acc_threshold']:.4f}")
            if 'accuracy' in metrics_to_print:
                print(f"Accuracy:                    {metrics_to_print['accuracy']:.4f}")
            if 'roc_auc' in metrics_to_print:
                print(f"ROC AUC:                     {metrics_to_print['roc_auc']:.4f}")
            if 'f1' in metrics_to_print:
                print(f"F1 Score:                    {metrics_to_print['f1']:.4f}")
            if 'precision' in metrics_to_print:
                print(f"Precision:                   {metrics_to_print['precision']:.4f}")
            if 'recall' in metrics_to_print:
                print(f"Recall:                      {metrics_to_print['recall']:.4f}")
            print("-----------------------------\n")
            # Print all other metrics
            for k, v in metrics_to_print.items():
                if k not in [
                    'youden_j', 'top_acc_threshold', 'accuracy', 'roc_auc', 'f1', 'precision', 'recall',
                    'threshold', 'best_accuracy', 'best_accuracy_threshold'
                ]:
                    print(f"{k}: {v}")
            return test_metrics
        else:
            print("[DEBUG] No best model found for final test set evaluation.")
            return self.results
    
    def _save_results(self, study):
        """Save optimization results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/optuna_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")