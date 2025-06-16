import torch.nn as nn
import torch.nn.functional as F

class EuclideanTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        # Normalize vectors to project onto the unit hypersphere
        anchor_norm = F.normalize(anchor, dim=1)
        positive_norm = F.normalize(positive, dim=1)
        negative_norm = F.normalize(negative, dim=1)

        # Cosine similarity: closer to 1 means more similar
        sim_pos = F.cosine_similarity(anchor_norm, positive_norm)
        sim_neg = F.cosine_similarity(anchor_norm, negative_norm)

        # Convert similarity to distance: 1 - cosine similarity
        dist_pos = 1 - sim_pos
        dist_neg = 1 - sim_neg

        # Apply triplet margin ranking on cosine distances
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()

class HybridTripletLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.5):
        super().__init__()
        self.margin = margin
        self.alpha = alpha  # controls the balance between cosine and euclidean

    def forward(self, anchor, positive, negative):
        anchor_norm = F.normalize(anchor, dim=1)
        positive_norm = F.normalize(positive, dim=1)
        negative_norm = F.normalize(negative, dim=1)

        # Cosine distances
        cos_pos = 1 - F.cosine_similarity(anchor_norm, positive_norm)
        cos_neg = 1 - F.cosine_similarity(anchor_norm, negative_norm)

        # Euclidean distances
        euc_pos = F.pairwise_distance(anchor, positive, p=2)
        euc_neg = F.pairwise_distance(anchor, negative, p=2)

        # Combined distances
        dist_pos = self.alpha * cos_pos + (1 - self.alpha) * euc_pos
        dist_neg = self.alpha * cos_neg + (1 - self.alpha) * euc_neg

        # Triplet loss
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean() 