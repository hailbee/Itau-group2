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
    def __init__(self, model, criterion, optimizer, device, log_csv_path="training_log.csv", model_type=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_csv_path = log_csv_path
        self.model_type = model_type
        self.model.to(device)
        self.evaluator = Evaluator(model, model_type=model_type)

    def train_epoch(self, dataloader, mode="pair"):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        for i, batch in enumerate(dataloader):
            # Handle different modes based on expected inputs
            if mode == "triplet":
                anchor_texts, positive_texts, negative_texts = batch
                outputs = self.model(anchor_texts, positive_texts, negative_texts)
            elif mode == "supcon":
                anchor_texts, positive_texts, negative_texts = batch
                outputs = self.model(anchor_texts, positive_texts, negative_texts)
            elif mode == "infonce":
                anchor_texts, positive_texts, negative_texts = batch
                outputs = self.model(anchor_texts, positive_texts, negative_texts)
            else:  # pair mode
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

    # pass in curriculum learning parameter 
    def train(self, dataloader, test_reference_filepath, test_filepath, 
             mode="pair", epochs=30, warmup_loader=None, warmup_epochs=5, curriculum = None, validate_filepath=None):
        """
        Main training loop with optional warmup and validation.
        
        Args:
            dataloader: Main training dataloader
            test_reference_filepath: Path to reference test data
            test_filepath: Path to test data
            mode: "pair" or "triplet"
            epochs: Number of training epochs
            warmup_loader: Optional warmup dataloader
            warmup_epochs: Number of warmup epochs
            validate_filepath: Path to validation data file (optional)
            
        Returns:
            dict: Best metrics achieved during training (including validation)
        """
        # bandit learning setup - only if warmup_loader is provided
        if warmup_loader is not None:
            datasets = {
                "easy": warmup_loader.dataset,
                "hard": dataloader.dataset
            }
            # bandit learning tracking
            rewards = {k: [] for k in datasets}
        else:
            datasets = {}
            rewards = {}
        
        # keeping track of accuracy
        prev_accuracy = 0.0
        best_metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'roc_auc': 0.0
        }
        best_epochs = {
            'accuracy': -1,
            'precision': -1,
            'recall': -1,
            'roc_auc': -1
        }
        best_epoch_loss = float('inf')

        best_val_metrics = None
        best_val_epoch = -1
        val_metrics_at_halfway = None
        halfway_epoch = epochs // 2

        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "AUC", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_AUC"])

            for epoch in range(epochs):
                
                # self paced curriculum learning - only if warmup_loader is provided
                if curriculum == "self" and warmup_loader is not None:
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
                
                # bandit curriculum learning - only if warmup_loader is provided
                elif curriculum == "bandit" and warmup_loader is not None:

                    # exploration rate
                    epsilon = 0.1 
                    reward_window = 5
                    
                    best_metrics = {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'roc_auc': 0.0
                    }
                    best_epochs = {
                        'accuracy': -1,
                        'precision': -1,
                        'recall': -1,
                        'roc_auc': -1
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
                    print(mix_desc)

                else:
                    # non curriculum mode
                    current_loader = warmup_loader if warmup_loader and epoch < warmup_epochs else dataloader

                # Train epoch
                avg_loss = self.train_epoch(current_loader, mode)
                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

                if avg_loss < best_epoch_loss:
                    best_epoch_loss = avg_loss

                # Evaluate on validation set at halfway and end
                val_metrics = None
                if validate_filepath is not None and (epoch == halfway_epoch or epoch == epochs - 1):
                    print(f"[DEBUG] Evaluating on validation set at epoch {epoch+1}")
                    val_metrics = self.evaluator.evaluate(None, validate_filepath)
                    if epoch == halfway_epoch:
                        val_metrics_at_halfway = val_metrics
                    # Track best validation metric (AUC)
                    if best_val_metrics is None or val_metrics['roc_auc'] > best_val_metrics['roc_auc']:
                        best_val_metrics = val_metrics
                        best_val_epoch = epoch + 1

                # Evaluate on test set (for logging)
                metrics = self.evaluate(test_reference_filepath, test_filepath)

                if curriculum == "bandit" and warmup_loader is not None:
                    delta_acc = metrics['accuracy'] - prev_accuracy
                    prev_accuracy = metrics['accuracy']
                    rewards[chosen_dataset_name].append(delta_acc)
                
                # Log metrics
                writer.writerow([
                    epoch + 1, 
                    avg_loss, 
                    metrics['accuracy'], 
                    metrics['precision'], 
                    metrics['recall'],
                    metrics.get('roc_auc', None),
                    val_metrics['accuracy'] if val_metrics else '',
                    val_metrics['precision'] if val_metrics else '',
                    val_metrics['recall'] if val_metrics else '',
                    val_metrics['roc_auc'] if val_metrics else ''
                ])

                print(f"Epoch {epoch+1} - Test Accuracy: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"AUC: {metrics.get('roc_auc', float('nan')):.4f}")
                if val_metrics:
                    print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f} | "
                          f"Precision: {val_metrics['precision']:.4f} | "
                          f"Recall: {val_metrics['recall']:.4f} | "
                          f"AUC: {val_metrics['roc_auc']:.4f}")

                # Track best metrics
                for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
                    if metric in metrics and metrics[metric] > best_metrics[metric]:
                        best_metrics[metric] = metrics[metric]
                        best_epochs[metric] = epoch + 1

        # Print summary
        print("\n=== Best Epochs Summary ===")
        for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
            if best_epochs[metric] != -1:
                print(f"Best {metric.capitalize()}: {best_metrics[metric]:.4f} "
                      f"at epoch {best_epochs[metric]}")
        print(f"Best Loss: {best_epoch_loss:.4f}")

        # Add loss and validation to best_metrics
        best_metrics['loss'] = best_epoch_loss
        if best_val_metrics:
            best_metrics['val_auc'] = best_val_metrics['roc_auc']
            best_metrics['val_accuracy'] = best_val_metrics['accuracy']
            best_metrics['val_precision'] = best_val_metrics['precision']
            best_metrics['val_recall'] = best_val_metrics['recall']
        if val_metrics_at_halfway:
            print(f"Validation metrics at halfway (epoch {halfway_epoch+1}): {val_metrics_at_halfway}")
        return best_metrics 