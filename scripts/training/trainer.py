import csv
import random
import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
from scripts.evaluation.evaluator import Evaluator


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


    def evaluate(self, test_filepath):
        """Evaluate model on test set (pairwise only)"""
        self.model.eval()
        _, metrics = self.evaluator.evaluate(test_filepath)
        return metrics


    def train(self, dataloader, test_filepath, 
             mode="pair", epochs=30, warmup_loader=None, warmup_epochs=5, curriculum = None, validate_filepath=None):
        """
        Main training loop with optional warmup and validation.
        """
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
        halfway_epoch = (epochs - 1) // 2

        for epoch in range(epochs):

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

            elif curriculum == "bandit" and warmup_loader is not None:
                epsilon = 0.1
                reward_window = 5

                # Bandit curriculum learning
                # For simplicity, use a local rewards dict
                rewards = {"easy": [], "hard": []}

                if random.random() < epsilon:
                    chosen_dataset_name = random.choice(["easy", "hard"])
                else:
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
                current_loader = warmup_loader if warmup_loader and epoch < warmup_epochs else dataloader

            avg_loss = self.train_epoch(current_loader, mode)
            print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss

            val_metrics = None
            if validate_filepath is not None and (epoch == halfway_epoch or epoch == epochs - 1):
                print(f"[DEBUG] Evaluating on validation set at epoch {epoch+1}")
                val_metrics = self.evaluator.evaluate(validate_filepath)[1]

                if epoch == halfway_epoch:
                    val_metrics_at_halfway = val_metrics

                if best_val_metrics is None or (val_metrics and val_metrics.get('roc_auc', 0) > best_val_metrics.get('roc_auc', 0)):
                    best_val_metrics = val_metrics
                    best_val_epoch = epoch + 1

            if val_metrics:
                print(f"Epoch {epoch+1} - Val Accuracy: {val_metrics['accuracy']:.4f} | "
                        f"Precision: {val_metrics['precision']:.4f} | "
                        f"Recall: {val_metrics['recall']:.4f} | "
                        f"AUC: {val_metrics['roc_auc']:.4f}")

        print("\n=== Best Epochs Summary ===")
        for metric in ['accuracy', 'precision', 'recall', 'roc_auc']:
            if best_metrics[metric] < best_val_metrics.get(metric, 0):
                best_metrics[metric] = best_val_metrics.get(metric, 0)
                best_epochs[metric] = best_val_epoch
                print(f"Best {metric.capitalize()}: {best_metrics[metric]:.4f} at epoch {best_epochs[metric]}")

        print(f"Best Loss: {best_epoch_loss:.4f}")

        best_metrics['loss'] = best_epoch_loss
        if best_val_metrics:
            best_metrics['val_auc'] = best_val_metrics.get('roc_auc', None)
            best_metrics['val_accuracy'] = best_val_metrics.get('accuracy', None)
            best_metrics['val_precision'] = best_val_metrics.get('precision', None)
            best_metrics['val_recall'] = best_val_metrics.get('recall', None)
            best_metrics['val_epoch'] = best_val_epoch
        if val_metrics_at_halfway:
            pass # Removed print statement

        return best_metrics 