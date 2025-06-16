from .base import BaseSiameseCLIP

class SiameseCLIPModelPairs(BaseSiameseCLIP):
    """
    Siamese network for pair-wise learning using CLIP as backbone.
    """
    def forward(self, text1, text2):
        z1 = self.encode(text1)
        z2 = self.encode(text2)
        return z1, z2

class SiameseCLIPTriplet(BaseSiameseCLIP):
    """
    Siamese network for triplet learning using CLIP as backbone.
    """
    def forward(self, anchor_texts, positive_texts, negative_texts):
        z_anchor = self.encode(anchor_texts)
        z_positive = self.encode(positive_texts)
        z_negative = self.encode(negative_texts)
        return z_anchor, z_positive, z_negative 