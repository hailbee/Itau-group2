### Retrieving embeddings from trained model

class EmbeddingExtractor(nn.Module):
    def __init__(self, siamese_model):
        super().__init__()
        self.clip = siamese_model.clip
        self.projector = siamese_model.projector

    def forward(self, texts):
        inputs = clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.projector[0].weight.device)
        with torch.no_grad():
            features = self.clip.get_text_features(**inputs)
        projected = self.projector(features)
        return F.normalize(projected, dim=1)