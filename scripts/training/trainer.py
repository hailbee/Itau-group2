import torch
import csv
from sklearn.metrics import precision_score, recall_score, roc_curve
from utils.evals import find_best_threshold_youden
from scripts.evaluation.evaluator import Evaluator
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import random
from model_utils.loss.supcon_loss import SupConLoss
from model_utils.loss.infonce_loss import InfoNCELoss

class Trainer:
    """
    Unified training interface for both pair and triplet models.
    Handles training, validation, and logging.
    """
    def __init__(self, model, criterion, optimizer, device, log_csv_path="training_log.csv"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model.to(device)
        self.evaluator = Evaluator(model)

    def train_epoch(self, dataloader, mode="pair"):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # Unified logic: model and criterion handle all modes
            outputs = self.model(*batch)
            loss = self.criterion(*outputs)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")

        return epoch_loss / len(dataloader)

    def evaluate(self, test_reference_filepath, test_filepath):
        """Evaluate model on test set"""
        self.model.eval()
        results_df, metrics = self.evaluator.evaluate(test_reference_filepath, test_filepath)
        return metrics

    # pass in cirriculum learning parameter 
    def train(self, dataloader, test_reference_filepath, test_filepath, 
             mode="pair", epochs=30, warmup_loader=None, warmup_epochs=5, cirriculum = None):
        """
        Main training loop with optional warmup.
        
        Args:
            dataloader: Main training dataloader
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair" or "triplet"
            epochs: Number of training epochs
            warmup_loader: Optional warmup dataloader
            warmup_epochs: Number of warmup epochs
        """
        # bandit learning setup 
        datasets = {
            "easy": warmup_loader.dataset,
            "hard": dataloader.dataset
        }

        # bandit learning tracking
        rewards = {k: [] for k in datasets}
        # keeping track of accuracy
        prev_accuracy = 0.0

        best_epoch_loss = float('inf')

        # bandit learning metrics
        best_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0}
        best_epochs = {'accuracy': -1, 'precision': -1, 'recall': -1}

        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall"])

            for epoch in range(epochs):
                
                # self paced cirriculum learning
                if cirriculum == "self":
                    hard_ratio = min(0.1 * epoch, 1.0)
                    easy_ratio = 1.0 - hard_ratio

                    total_samples = len(dataloader.dataset)
                    num_easy = int(total_samples * easy_ratio)
                    num_hard = total_samples - num_easy

                    easy_indices = np.random.choice(len(warmup_loader.dataset), num_easy, replace=False)
                    hard_indices = np.random.choice(len(dataloader.dataset), num_hard, replace=False)

                    mixed_dataset = ConcatDataset([
                        Subset(warmup_loader.dataset, easy_indices),
                        Subset(dataloader.dataset, hard_indices)
                    ])

                    current_loader = DataLoader(mixed_dataset, batch_size=dataloader.batch_size, shuffle=True)
                
                # bandit cirriculum learning
                elif cirriculum == "bandit":

                    # exploration rate
                    epsilon = 0.1 
                    reward_window = 5
                    
                    best_metrics = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0
                    }
                    best_epochs = {
                        'accuracy': -1,
                        'precision': -1,
                        'recall': -1
                    }

                     # Bandit curriculum learning
                    if random.random() < epsilon:
                        chosen_dataset_name = random.choice(["easy", "hard"])
                    else:
                        # reward estimation
                        avg_rewards = {
                            k: np.mean(v[-reward_window:]) if v else 0.0
                            for k, v in rewards.items()
                        }
                        chosen_dataset_name = max(avg_rewards, key=avg_rewards.get)

                    easy_batch_size = dataloader.batch_size // 2 if chosen_dataset_name == "hard" else dataloader.batch_size
                    hard_batch_size = dataloader.batch_size - easy_batch_size

                    easy_indices = np.random.choice(len(warmup_loader.dataset), easy_batch_size, replace=False)
                    hard_indices = np.random.choice(len(dataloader.dataset), hard_batch_size, replace=False)

                    mixed_dataset = ConcatDataset([
                        Subset(warmup_loader.dataset, easy_indices),
                        Subset(dataloader.dataset, hard_indices)
                    ])

                    current_loader = DataLoader(mixed_dataset, batch_size=dataloader.batch_size, shuffle=True)
                    mix_desc = f"bandit (chose {chosen_dataset_name})"

                else:
                    # non cirriculum mode
                    current_loader = warmup_loader if warmup_loader and epoch < warmup_epochs else dataloader

                # Train 
                avg_loss = self.train_epoch(current_loader, mode)
                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

                if avg_loss < best_epoch_loss:
                    best_epoch_loss = avg_loss

                # Evaluate
                metrics = self.evaluate(test_reference_filepath, test_filepath)

                if cirriculum == "bandit":
                    delta_acc = metrics['accuracy'] - prev_accuracy
                    prev_accuracy = metrics['accuracy']
                    rewards[chosen_dataset_name].append(delta_acc)
                
                # Log metrics
                writer.writerow([
                    epoch + 1, 
                    avg_loss, 
                    metrics['accuracy'], 
                    metrics['precision'], 
                    metrics['recall']
                ])

                print(f"Epoch {epoch+1} - Test Accuracy: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f}")

                # Track best metrics
                for metric in ['accuracy', 'precision', 'recall']:
                    if metrics[metric] > best_metrics[metric]:
                        best_metrics[metric] = metrics[metric]
                        best_epochs[metric] = epoch + 1

        # Print summary
        print("\n=== Best Epochs Summary ===")
        for metric in ['accuracy', 'precision', 'recall']:
            print(f"Best {metric.capitalize()}: {best_metrics[metric]:.4f} "
                  f"at epoch {best_epochs[metric]}")

        return best_epoch_loss 