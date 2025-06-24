import torch
import pandas as pd
import numpy as np
import copy
import os
from datetime import datetime
from scripts.optimization.bayesian import BayesianOptimizer
from scripts.optimization.random_optimizer import RandomOptimizer
from scripts.optimization.optuna import OptunaOptimizer
from scripts.optimization.pbt import PopulationBasedTrainer
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator

class UnifiedHyperparameterOptimizer:
    """
    Unified interface for different hyperparameter optimization methods.
    Supports Bayesian optimization, random search, Optuna, and Population-Based Training.
    """
    
    def __init__(self, model_class, device, log_dir="optimization_results"):
        self.model_class = model_class
        self.device = device
        self.log_dir = log_dir
        self.results = []
        self.best_auc = 0.0  # Track best AUC across all trials
        
        # Create main log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize different optimizers
        self.bayesian_optimizer = BayesianOptimizer(model_class, device, f"{log_dir}/bayesian")
        self.random_optimizer = RandomOptimizer(model_class, device, f"{log_dir}/random")
        self.optuna_optimizer = OptunaOptimizer(model_class, device, f"{log_dir}/optuna")
        self.pbt_optimizer = PopulationBasedTrainer(model_class, device, f"{log_dir}/pbt")
    
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
    
    def create_optimizer(self, model, params):
        """Create optimizer based on parameters"""
        if params['optimizer'] == "adam":
            return torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        elif params['optimizer'] == "adamw":
            return torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        else:  # sgd
            return torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    
    def train_population(self, population, reference_filepath, test_reference_filepath, test_filepath,
                        mode, loss_type, warmup_filepath=None, epochs_per_generation=5, 
                        warmup_epochs=5, generations=10, evolution_frequency=2):
        """
        Train the population with evolution.
        
        Args:
            population: List of hyperparameter dictionaries
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            warmup_filepath: Optional warmup data path
            epochs_per_generation: Number of epochs per generation
            warmup_epochs: Number of warmup epochs
            generations: Number of generations
            evolution_frequency: How often to evolve (every N generations)
        """
        print(f"Starting PBT with population size {len(population)}")
        print(f"Will run for {generations} generations with {epochs_per_generation} epochs each")
        
        # Load data
        dataframe = pd.read_pickle(reference_filepath)
        warmup_dataframe = None
        if warmup_filepath:
            warmup_dataframe = pd.read_pickle(warmup_filepath)
        
        # Initialize population models
        models = []
        optimizers = []
        trainers = []
        evaluators = []
        dataloaders = []
        warmup_loaders = []
        
        for i, params in enumerate(population):
            # Create dataloader
            dataloader = self.create_dataloader(dataframe, params['batch_size'], mode)
            dataloaders.append(dataloader)
            
            warmup_loader = None
            if warmup_dataframe is not None:
                warmup_loader = self.create_dataloader(warmup_dataframe, params['batch_size'], mode)
            warmup_loaders.append(warmup_loader)
            
            # Create model
            model = self.model_class(
                embedding_dim=512,
                projection_dim=params['internal_layer_size']
            ).to(self.device)
            models.append(model)
            
            # Create optimizer
            optimizer = self.create_optimizer(model, params)
            optimizers.append(optimizer)
            
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
                log_csv_path=f"{self.log_dir}/pbt_training_log_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            trainers.append(trainer)
            
            evaluator = Evaluator(model, batch_size=params['batch_size'])
            evaluators.append(evaluator)
        
        # Training loop with evolution
        for generation in range(generations):
            print(f"\n=== Generation {generation + 1}/{generations} ===")
            
            # Train all models in the population
            generation_results = []
            for i, (model, trainer, dataloader, warmup_loader) in enumerate(zip(models, trainers, dataloaders, warmup_loaders)):
                print(f"Training model {i+1}/{len(models)} with params: {population[i]}")
                
                # Train model
                model_loss = trainer.train(
                    dataloader=dataloader,
                    test_reference_filepath=test_reference_filepath,
                    test_filepath=test_filepath,
                    mode=mode,
                    epochs=epochs_per_generation,
                    warmup_loader=warmup_loader,
                    warmup_epochs=warmup_epochs if generation == 0 else 0
                )
                
                # Evaluate model
                results_df, metrics = evaluators[i].evaluate(
                    test_reference_filepath,
                    test_filepath
                )
                
                # Log results
                result = {
                    "trial_number": len(self.results) + 1,
                    "timestamp": datetime.now(),
                    "lr": population[i]['lr'],
                    "batch_size": population[i]['batch_size'],
                    "internal_layer_size": population[i]['internal_layer_size'],
                    "optimizer": population[i]['optimizer'],
                    "weight_decay": population[i]['weight_decay'],
                    "epochs": epochs_per_generation,
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
                    print(f"Trial {len(self.results)}: New best AUC = {self.best_auc:.4f}")
                else:
                    print(f"Trial {len(self.results)}: AUC = {current_auc:.4f} (Best = {self.best_auc:.4f})")
                
                result = {
                    "generation": generation + 1,
                    "model_id": i,
                    "timestamp": datetime.now(),
                    "accuracy": metrics['accuracy'],
                    "train_loss": model_loss,
                    "test_auc": metrics['roc_auc'],
                    "threshold": metrics['threshold'],
                    **population[i]
                }
                generation_results.append(result)
            
            # Evolution step
            if (generation + 1) % evolution_frequency == 0 and generation < generations - 1:
                print(f"\n--- Evolution step ---")
                self._evolve_population(models, population, generation_results, mode)
                
                # Update dataloaders for new batch sizes
                for i, params in enumerate(population):
                    dataloaders[i] = self.create_dataloader(dataframe, params['batch_size'], mode)
                    if warmup_dataframe is not None:
                        warmup_loaders[i] = self.create_dataloader(warmup_dataframe, params['batch_size'], mode)
                    
                    # Update optimizer for new parameters
                    optimizers[i] = self.create_optimizer(models[i], params)
                    
                    # Update trainer with new optimizer
                    loss_class = self.get_loss_class(mode, loss_type)
                    if mode in ["supcon", "infonce"]:
                        criterion = loss_class(temperature=params['temperature'])
                    elif mode == "triplet" and loss_type == "hybrid":
                        criterion = loss_class(margin=params['margin'], alpha=0.5)
                    else:
                        criterion = loss_class(margin=params['margin'])
                    
                    trainers[i] = Trainer(
                        model=models[i],
                        criterion=criterion,
                        optimizer=optimizers[i],
                        device=self.device,
                        log_csv_path=f"{self.log_dir}/pbt_training_log_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    )
        
        # Find best model
        best_result = max(self.results, key=lambda x: x['accuracy'])
        best_config = {k: v for k, v in best_result.items() if k not in ['generation', 'model_id', 'timestamp', 'accuracy', 'train_loss', 'test_auc', 'threshold']}
        best_config['best_accuracy'] = best_result['accuracy']
        
        print(f"\nPBT completed!")
        print(f"Best accuracy: {best_result['accuracy']:.4f}")
        print(f"Best configuration: {best_config}")
        
        # After optimization, print best AUC if available
        if isinstance(results_df, pd.DataFrame) and not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")
        
        return best_config, pd.DataFrame(self.results)
    
    def _evolve_population(self, models, population, generation_results, mode):
        """
        Evolve the population based on performance.
        
        Args:
            models: List of model objects
            population: List of hyperparameter dictionaries
            generation_results: Results from current generation
            mode: Training mode
        """
        # Sort by performance
        sorted_results = sorted(generation_results, key=lambda x: x['accuracy'], reverse=True)
        
        # Keep top 50% and replace bottom 50%
        keep_count = len(population) // 2
        
        for i in range(keep_count, len(population)):
            # Select a parent from top performers
            parent_idx = np.random.randint(0, keep_count)
            parent_result = sorted_results[parent_idx]
            parent_model_idx = parent_result['model_id']
            
            # Copy parent's model weights
            models[i].load_state_dict(models[parent_model_idx].state_dict())
            
            # Mutate parent's hyperparameters
            population[i] = self.mutate_hyperparameters(population[parent_model_idx], mode)
            
            print(f"Model {i+1} evolved from model {parent_model_idx+1} with new params: {population[i]}")
    
    def optimize(self, method, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None, **kwargs):
        """
        Perform hyperparameter optimization using the specified method.
        
        Args:
            method: Optimization method ("bayesian", "random", "optuna", "pbt")
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair", "triplet", "supcon", or "infonce"
            loss_type: Type of loss function to use
            warmup_filepath: Optional path to warmup data
            **kwargs: Additional arguments specific to each method
            
        Returns:
            tuple: (best_config, results_df, additional_info)
        """
        print(f"Starting {method} hyperparameter optimization...")
        
        if method == "bayesian":
            return self._run_bayesian_optimization(
                reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, **kwargs
            )
        elif method == "random":
            return self._run_random_optimization(
                reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, **kwargs
            )
        elif method == "optuna":
            return self._run_optuna_optimization(
                reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, **kwargs
            )
        elif method == "pbt":
            return self._run_pbt_optimization(
                reference_filepath, test_reference_filepath, test_filepath,
                mode, loss_type, warmup_filepath, **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
    
    def _run_bayesian_optimization(self, reference_filepath, test_reference_filepath, test_filepath,
                                 mode, loss_type, warmup_filepath, **kwargs):
        """Run Bayesian optimization"""
        n_calls = kwargs.get('n_calls', 50)
        n_random_starts = kwargs.get('n_random_starts', 10)
        epochs = kwargs.get('epochs', 5)
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        
        best_config, results_df = self.bayesian_optimizer.optimize(
            reference_filepath=reference_filepath,
            test_reference_filepath=test_reference_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            warmup_filepath=warmup_filepath,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            n_calls=n_calls,
            n_random_starts=n_random_starts
        )
        
        # After optimization, print best AUC if available
        if isinstance(results_df, pd.DataFrame) and not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")
        
        return best_config, results_df, {"method": "bayesian"}
    
    def _run_random_optimization(self, reference_filepath, test_reference_filepath, test_filepath,
                               mode, loss_type, warmup_filepath, **kwargs):
        """Run random search optimization"""
        n_trials = kwargs.get('n_trials', 50)
        epochs = kwargs.get('epochs', 5)
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        
        best_config, results_df = self.random_optimizer.optimize(
            reference_filepath=reference_filepath,
            test_reference_filepath=test_reference_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            warmup_filepath=warmup_filepath,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            n_trials=n_trials
        )
        
        # After optimization, print best AUC if available
        if isinstance(results_df, pd.DataFrame) and not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")
        
        return best_config, results_df, {"method": "random"}
    
    def _run_optuna_optimization(self, reference_filepath, test_reference_filepath, test_filepath,
                               mode, loss_type, warmup_filepath, **kwargs):
        """Run Optuna optimization"""
        n_trials = kwargs.get('n_trials', 50)
        epochs = kwargs.get('epochs', 5)
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        sampler = kwargs.get('sampler', 'tpe')
        pruner = kwargs.get('pruner', 'median')
        study_name = kwargs.get('study_name', None)
        
        best_config, results_df, study = self.optuna_optimizer.optimize(
            reference_filepath=reference_filepath,
            test_reference_filepath=test_reference_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            warmup_filepath=warmup_filepath,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
            n_trials=n_trials,
            sampler=sampler,
            pruner=pruner,
            study_name=study_name
        )
        
        # Create visualizations if requested
        if kwargs.get('create_visualizations', True):
            try:
                self.optuna_optimizer.visualize_results(study, self.log_dir)
            except Exception as e:
                print(f"Warning: Could not create visualizations: {e}")
        
        # After optimization, print best AUC if available
        if isinstance(results_df, pd.DataFrame) and not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")
        
        return best_config, results_df, {"method": "optuna", "study": study}
    
    def _run_pbt_optimization(self, reference_filepath, test_reference_filepath, test_filepath,
                            mode, loss_type, warmup_filepath, **kwargs):
        """Run Population-Based Training optimization"""
        epochs_per_generation = kwargs.get('epochs_per_generation', 5)
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        generations = kwargs.get('generations', 10)
        population_size = kwargs.get('population_size', 8)
        evolution_frequency = kwargs.get('evolution_frequency', 2)
        
        best_config, results_df = self.pbt_optimizer.optimize(
            reference_filepath=reference_filepath,
            test_reference_filepath=test_reference_filepath,
            test_filepath=test_filepath,
            mode=mode,
            loss_type=loss_type,
            warmup_filepath=warmup_filepath,
            epochs_per_generation=epochs_per_generation,
            warmup_epochs=warmup_epochs,
            generations=generations,
            population_size=population_size,
            evolution_frequency=evolution_frequency
        )
        
        # After optimization, print best AUC if available
        if isinstance(results_df, pd.DataFrame) and not results_df.empty and 'test_auc' in results_df.columns:
            best_auc = results_df['test_auc'].max()
            best_auc_row = results_df.loc[results_df['test_auc'].idxmax()]
            print(f"Best AUC: {best_auc:.4f}")
            print(f"Best AUC parameters: {best_auc_row.to_dict()}")
        
        return best_config, results_df, {"method": "pbt"}
    
    def compare_methods(self, reference_filepath, test_reference_filepath, test_filepath,
                       mode="pair", loss_type="cosine", warmup_filepath=None, **kwargs):
        """
        Compare different optimization methods on the same problem.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            warmup_filepath: Optional warmup data path
            **kwargs: Additional arguments for optimization
            
        Returns:
            dict: Results from all methods
        """
        methods = ["bayesian", "random", "optuna"]
        results = {}
        
        for method in methods:
            print(f"\n{'='*50}")
            print(f"Running {method.upper()} optimization")
            print(f"{'='*50}")
            
            try:
                best_config, results_df, additional_info = self.optimize(
                    method, reference_filepath, test_reference_filepath, test_filepath,
                    mode, loss_type, warmup_filepath, **kwargs
                )
                
                results[method] = {
                    "best_config": best_config,
                    "results_df": results_df,
                    "additional_info": additional_info
                }
                
                print(f"{method.upper()} completed successfully!")
                print(f"Best accuracy: {best_config.get('best_accuracy', 'N/A')}")
                
            except Exception as e:
                print(f"Error in {method} optimization: {e}")
                results[method] = {"error": str(e)}
        
        # Save comparison results
        comparison_summary = {}
        for method, result in results.items():
            if "error" not in result:
                comparison_summary[method] = {
                    "best_accuracy": result["best_config"].get("best_accuracy", 0),
                    "best_params": {k: v for k, v in result["best_config"].items() 
                                  if k not in ["best_accuracy", "mode", "loss_type", "optimization_method"]}
                }
            else:
                comparison_summary[method] = {"error": result["error"]}
        
        # Save comparison
        comparison_df = pd.DataFrame(comparison_summary).T
        comparison_df.to_csv(f"{self.log_dir}/method_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        print(f"\n{'='*50}")
        print("METHOD COMPARISON SUMMARY")
        print(f"{'='*50}")
        for method, summary in comparison_summary.items():
            if "error" not in summary:
                print(f"{method.upper()}: {summary['best_accuracy']:.4f}")
            else:
                print(f"{method.upper()}: ERROR - {summary['error']}")
        
        return results
    
    def get_recommended_settings(self, mode, loss_type, dataset_size=None):
        """
        Get recommended settings for different optimization methods.
        
        Args:
            mode: Training mode
            loss_type: Loss function type
            dataset_size: Size of the dataset (optional)
            
        Returns:
            dict: Recommended settings for each method
        """
        recommendations = {
            "bayesian": {
                "n_calls": 50,
                "n_random_starts": 10,
                "epochs": 5,
                "description": "Good for continuous hyperparameters, efficient exploration"
            },
            "random": {
                "n_trials": 50,
                "epochs": 5,
                "description": "Simple and often effective, good baseline"
            },
            "optuna": {
                "n_trials": 50,
                "sampler": "tpe",
                "pruner": "median",
                "epochs": 5,
                "description": "Advanced optimization with pruning and visualization"
            },
            "pbt": {
                "population_size": 8,
                "generations": 10,
                "epochs_per_generation": 5,
                "evolution_frequency": 2,
                "description": "Dynamic optimization, good for long training runs"
            }
        }
        
        # Adjust based on dataset size
        if dataset_size:
            if dataset_size < 1000:
                # Small dataset - reduce trials
                for method in recommendations:
                    if "n_calls" in recommendations[method]:
                        recommendations[method]["n_calls"] = 20
                    if "n_trials" in recommendations[method]:
                        recommendations[method]["n_trials"] = 20
                    if "population_size" in recommendations[method]:
                        recommendations[method]["population_size"] = 4
            elif dataset_size > 10000:
                # Large dataset - increase trials
                for method in recommendations:
                    if "n_calls" in recommendations[method]:
                        recommendations[method]["n_calls"] = 100
                    if "n_trials" in recommendations[method]:
                        recommendations[method]["n_trials"] = 100
                    if "population_size" in recommendations[method]:
                        recommendations[method]["population_size"] = 16
        
        return recommendations 