import torch
import csv
from sklearn.metrics import precision_score, recall_score, roc_curve
from utils.evals import find_best_threshold_youden
from scripts.evaluation.evaluator import Evaluator
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
            if mode == "pair":
                text1, text2, label = batch
                label = label.to(self.device)
                z1, z2 = self.model(text1, text2)
                loss = self.criterion(z1, z2, label)
            elif isinstance(self.criterion, SupConLoss):
                anchor_text, positive_texts, negative_texts = batch
                z_anchor = self.model.encode(anchor_text)
                z_positives = torch.stack([self.model.encode(pos) for pos in positive_texts], dim=1)
                z_negatives = torch.stack([self.model.encode(neg) for neg in negative_texts], dim=1)
                loss = self.criterion(z_anchor, z_positives, z_negatives)
            elif isinstance(self.criterion, InfoNCELoss):
                anchor_text, positive_text, negative_texts = batch
                z_anchor = self.model.encode(anchor_text)
                z_positive = self.model.encode(positive_text)
                z_negatives = torch.stack([self.model.encode(neg) for neg in negative_texts], dim=1)
                # InfoNCE expects anchor, positives (as [batch, 1, emb]), negatives
                z_positives = z_positive.unsqueeze(1)  # [batch, 1, emb]
                loss = self.criterion(z_anchor, z_positives, z_negatives)
            else:
                anchor_text, positive_text, negative_text = batch
                z_anchor, z_positive, z_negative = self.model(anchor_text, positive_text, negative_text)
                loss = self.criterion(z_anchor, z_positive, z_negative)

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

    def train(self, dataloader, test_reference_filepath, test_filepath, 
             mode="pair", epochs=30, warmup_loader=None, warmup_epochs=5):
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
        best_epoch_loss = float('inf')

        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "F1"])

            for epoch in range(epochs):
                # Use warmup loader if provided and in warmup phase
                current_loader = warmup_loader if warmup_loader and epoch < warmup_epochs else dataloader
                
                # Train epoch
                avg_loss = self.train_epoch(current_loader, mode)
                print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

                if avg_loss < best_epoch_loss:
                    best_epoch_loss = avg_loss

                # Evaluate
                metrics = self.evaluate(test_reference_filepath, test_filepath)
                
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