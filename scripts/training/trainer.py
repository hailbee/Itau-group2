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
            # Add debugging for device placement
            print(f"DEBUG: Batch {i} - mode: {mode}")
            print(f"DEBUG: Model device: {next(self.model.parameters()).device}")
            print(f"DEBUG: Batch type: {type(batch)}")
            print(f"DEBUG: Batch length: {len(batch)}")
            
            if mode == "pair":
                text1, text2, labels = batch
                print(f"DEBUG: text1 type: {type(text1)}, text2 type: {type(text2)}")
                print(f"DEBUG: labels type: {type(labels)}, device: {labels.device if hasattr(labels, 'device') else 'N/A'}")
                # Move labels to device if it's a tensor
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)
                    print(f"DEBUG: Moved labels to device: {labels.device}")
            else:
                print(f"DEBUG: Non-pair mode, batch contents: {[type(item) for item in batch]}")
            
            # Unified logic: model and criterion handle all modes
            outputs = self.model(*batch)
            print(f"DEBUG: Model outputs type: {type(outputs)}")
            if isinstance(outputs, tuple):
                print(f"DEBUG: Outputs devices: {[out.device if hasattr(out, 'device') else 'N/A' for out in outputs]}")
            
            loss = self.criterion(*outputs)
            print(f"DEBUG: Loss computed: {loss}, device: {loss.device}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 100 == 0:
                print(f"Step {i} complete out of {len(dataloader)}")
            
            # Only debug first few batches
            if i >= 2:
                break

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
            
        Returns:
            dict: Best metrics achieved during training
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

        with open(self.log_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "Accuracy", "Precision", "Recall", "AUC"])

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
                    metrics['recall'],
                    metrics.get('roc_auc', None)
                ])

                print(f"Epoch {epoch+1} - Test Accuracy: {metrics['accuracy']:.4f} | "
                      f"Precision: {metrics['precision']:.4f} | "
                      f"Recall: {metrics['recall']:.4f} | "
                      f"AUC: {metrics.get('roc_auc', float('nan')):.4f}")

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

        # Add loss to best_metrics
        best_metrics['loss'] = best_epoch_loss

        return best_metrics 