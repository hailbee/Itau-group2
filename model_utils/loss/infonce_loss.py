import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss that works with anchor, positives, and negatives format.
    
    Args:
        temperature (float): Temperature parameter for scaling the similarity scores
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, negatives):
        """
        Forward pass computing the loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positives: List of positive embeddings [batch_size, n_positives, embedding_dim]
            negatives: List of negative embeddings [batch_size, n_negatives, embedding_dim]
            
        Returns:
            torch.Tensor: Mean loss value
        """
        device = anchor.device
        batch_size = anchor.shape[0]
        
        # Normalize all embeddings
        anchor = F.normalize(anchor, dim=1)
        positives = F.normalize(positives, dim=2)
        negatives = F.normalize(negatives, dim=2)

        print(f"[DEBUG] InfoNCELoss forward: anchor shape={anchor.shape}, positives shape={positives.shape}, negatives shape={negatives.shape}")
        
        # Compute similarities between anchor and positives
        pos_similarities = torch.bmm(
            anchor.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            positives.transpose(1, 2)  # [batch_size, embedding_dim, n_positives]
        ).squeeze(1) / self.temperature  # [batch_size, n_positives]
        
        # Compute similarities between anchor and negatives
        neg_similarities = torch.bmm(
            anchor.unsqueeze(1),  # [batch_size, 1, embedding_dim]
            negatives.transpose(1, 2)  # [batch_size, embedding_dim, n_negatives]
        ).squeeze(1) / self.temperature  # [batch_size, n_negatives]
        
        # Concatenate positive and negative similarities
        all_similarities = torch.cat([pos_similarities, neg_similarities], dim=1)
        
        # Create labels: 0 for positives (first n_positives columns), 1 for negatives
        labels = torch.zeros(batch_size, device=device, dtype=torch.long)
        
        # Compute cross entropy loss
        loss = F.cross_entropy(all_similarities, labels)
        
        return loss 