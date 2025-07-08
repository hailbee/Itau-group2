from .base import BaseSiameseCLIP

class SiameseCLIPModelPairs(BaseSiameseCLIP):
    """
    Siamese network for pair-wise learning using CLIP as backbone.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, freeze_clip=True, backbone=None, tokenizer=None):
        super().__init__(embedding_dim, projection_dim, freeze_clip, backbone, tokenizer)
    def forward(self, text1, text2, label=None):
        z1 = self.encode(text1)
        z2 = self.encode(text2)
        # Move label to the same device as z1 if it's a tensor
        if label is not None and hasattr(label, 'device'):
            label = label.to(z1.device)
        return z1, z2, label

class SiameseCLIPTriplet(BaseSiameseCLIP):
    """
    Siamese network for triplet learning using CLIP as backbone.
    """
    def __init__(self, embedding_dim=512, projection_dim=128, freeze_clip=True, backbone=None, tokenizer=None):
        super().__init__(embedding_dim, projection_dim, freeze_clip, backbone, tokenizer)
    def forward(self, anchor_texts, positive_texts, negative_texts):
        z_anchor = self.encode(anchor_texts)
        z_positive = self.encode(positive_texts)
        z_negative = self.encode(negative_texts)
        return z_anchor, z_positive, z_negative
