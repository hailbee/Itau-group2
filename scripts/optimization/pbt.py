import torch
import pandas as pd
import numpy as np
from datetime import datetime
import copy
import os
from scripts.training.trainer import Trainer
from scripts.evaluation.evaluator import Evaluator
from .base_optimizer import BaseOptimizer

class PopulationBasedTrainer(BaseOptimizer):
    """
    Population-Based Training (PBT) for dynamic hyperparameter optimization.
    Maintains a population of models and evolves hyperparameters during training.
    """
    
    def __init__(self, model_type, model_name=None, device=None, log_dir="pbt_optimization_results"):
        super().__init__(model_type, model_name, device, log_dir)
        
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
            # Convert numpy types to Python types
            batch_size = int(params['batch_size'])
            internal_layer_size = int(params['internal_layer_size'])
            
            # Create dataloader
            dataloader = self.create_dataloader(dataframe, batch_size, mode)
            dataloaders.append(dataloader)
            
            warmup_loader = None
            if warmup_dataframe is not None:
                warmup_loader = self.create_dataloader(warmup_dataframe, batch_size, mode)
            warmup_loaders.append(warmup_loader)
            
            # Create model
            model = self.create_siamese_model(mode, internal_layer_size).to(self.device)
            models.append(model)
            
            # Create optimizer
            optimizer = self.create_optimizer(model, params)
            optimizers.append(optimizer)
            
            # Create trainer
            trainer = Trainer(
                model=model,
                criterion=None,  # Will be set during training
                optimizer=optimizer,
                device=self.device,
                log_csv_path=f"{self.log_dir}/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
            trainers.append(trainer)
            
            # Create evaluator
            evaluator = Evaluator(model, batch_size=batch_size)
            evaluators.append(evaluator)
        
        # Training loop with evolution
        for generation in range(generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{generations}")
            print(f"{'='*60}")
            
            # Train each member of the population
            generation_results = []
            for i, (model, optimizer, trainer, dataloader, warmup_loader) in enumerate(
                zip(models, optimizers, trainers, dataloaders, warmup_loaders)):
                
                print(f"\n{'='*50}")
                print(f"Training Population Member {i+1}/{len(population)}")
                print(f"{'='*50}")
                
                # Log parameters for this population member
                params = population[i]
                param_str = f"LR: {params['lr']:.6f}, Batch: {params['batch_size']}, Layer: {params['internal_layer_size']}, Opt: {params['optimizer']}, WD: {params['weight_decay']:.6f}"
                if mode in ["supcon", "infonce"]:
                    param_str += f", Temp: {params['temperature']:.4f}"
                else:
                    param_str += f", Margin: {params['margin']:.4f}"
                print(f"Member {i+1} parameters: {param_str}")
                
                # Get loss class and create criterion
                loss_class = self.get_loss_class(mode, loss_type)
                if mode in ["supcon", "infonce"]:
                    temperature = float(population[i]['temperature'])
                    criterion = loss_class(temperature=temperature)
                elif mode == "triplet" and loss_type == "hybrid":
                    margin = float(population[i]['margin'])
                    criterion = loss_class(margin=margin, alpha=0.5)
                else:
                    margin = float(population[i]['margin'])
                    criterion = loss_class(margin=margin)
                
                # Update trainer criterion
                trainer.criterion = criterion
                
                # Train model
                best_metrics = trainer.train(
                    dataloader=dataloader,
                    test_reference_filepath=test_reference_filepath,
                    test_filepath=test_filepath,
                    mode=mode,
                    epochs=epochs_per_generation,
                    warmup_loader=warmup_loader,
                    warmup_epochs=warmup_epochs
                )
                
                # Store results
                result = {
                    "generation": generation + 1,
                    "population_member": i + 1,
                    "timestamp": datetime.now(),
                    **population[i],
                    "mode": mode,
                    "loss_type": loss_type,
                    "model_type": self.model_type,
                    "model_name": self.model_name,
                    **best_metrics
                }
                generation_results.append(result)
                self.results.append(result)
                
                # Update best metrics
                if best_metrics.get('test_auc', 0) > self.best_auc:
                    self.best_auc = best_metrics['test_auc']
                if best_metrics.get('test_accuracy', 0) > self.best_accuracy:
                    self.best_accuracy = best_metrics['test_accuracy']
                
                print(f"\nMember {i+1} completed - AUC: {best_metrics.get('test_auc', 0):.4f}, Accuracy: {best_metrics.get('test_accuracy', 0):.4f}")
            
            # Evolve population if needed
            if (generation + 1) % evolution_frequency == 0 and generation < generations - 1:
                print(f"\nEvolving population at generation {generation + 1}")
                self._evolve_population(models, population, generation_results, mode)
        
        # Save results
        self._save_results()
        
        print(f"\n{'='*60}")
        print(f"PBT completed!")
        print(f"Best AUC: {self.best_auc:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"{'='*60}")
        
        return self.results
    
    def _evolve_population(self, models, population, generation_results, mode):
        """
        Evolve the population by replacing poor performers with mutated versions of good performers.
        
        Args:
            models: List of model instances
            population: List of hyperparameter dictionaries
            generation_results: List of results from current generation
            mode: Training mode
        """
        # Sort by performance (AUC)
        sorted_indices = sorted(
            range(len(generation_results)),
            key=lambda i: generation_results[i].get('test_auc', 0),
            reverse=True
        )
        
        # Keep top 50% of performers
        keep_count = len(population) // 2
        
        for i in range(keep_count, len(population)):
            # Select a good performer to copy from
            source_idx = sorted_indices[np.random.randint(keep_count)]
            
            # Copy model weights
            models[i].load_state_dict(models[source_idx].state_dict())
            
            # Mutate hyperparameters
            population[i] = self.mutate_hyperparameters(population[source_idx], mode)
            
            print(f"Evolved member {i+1} from member {source_idx+1}")
    
    def optimize(self, reference_filepath, test_reference_filepath, test_filepath,
                mode="pair", loss_type="cosine", warmup_filepath=None,
                epochs_per_generation=5, warmup_epochs=5, generations=10,
                population_size=8, evolution_frequency=2):
        """
        Run PBT optimization.
        
        Args:
            reference_filepath: Path to training data
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: Training mode
            loss_type: Loss function type
            warmup_filepath: Optional warmup data path
            epochs_per_generation: Number of epochs per generation
            warmup_epochs: Number of warmup epochs
            generations: Number of generations
            population_size: Size of the population
            evolution_frequency: How often to evolve (every N generations)
        """
        print(f"Starting PBT optimization for {self.model_type} model")
        print(f"Mode: {mode}, Loss: {loss_type}")
        print(f"Population size: {population_size}, Generations: {generations}")
        
        # Sample initial population
        population = self.sample_initial_hyperparameters(mode, population_size)
        
        # Train population
        return self.train_population(
            population, reference_filepath, test_reference_filepath, test_filepath,
            mode, loss_type, warmup_filepath, epochs_per_generation, 
            warmup_epochs, generations, evolution_frequency
        )
    
    def _save_results(self):
        """Save optimization results to CSV."""
        if self.results:
            df = pd.DataFrame(self.results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.log_dir}/pbt_results_{timestamp}.csv"
            df.to_csv(filename, index=False)
            print(f"Results saved to {filename}") 