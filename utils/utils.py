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

def batched_embedding(extractor, names, batch_size=32):
    embeddings = []
    for i in range(0, len(names), batch_size):
        batch = names[i:i+batch_size]
        emb = extractor(batch)
        embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0)

def plot_roc_curve(results_df):
    """
    Plots the ROC curve given a DataFrame with 'label' and 'max_similarity' columns.

    Args:
        results_df (pd.DataFrame): DataFrame containing 'label' and 'max_similarity' columns.
    Returns:
        roc_auc (float): Computed AUC value
        thresholds (np.ndarray): Array of threshold values
    """
    y_true = results_df['label']
    y_scores = results_df['max_similarity']

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return roc_auc, fpr, tpr, thresholds

def find_best_threshold_youden(fpr, tpr, thresholds):
    """
    Finds the best threshold based on Youden's J statistic (TPR - FPR).

    Args:
        fpr (np.ndarray): False positive rates.
        tpr (np.ndarray): True positive rates.
        thresholds (np.ndarray): Thresholds used to compute fpr and tpr.

    Returns:
        float: Best threshold maximizing TPR - FPR.
    """
    youden_index = tpr - fpr
    best_idx = youden_index.argmax()
    best_threshold = thresholds[best_idx]
    print(f"Best threshold (Youden): {best_threshold:.3f}")
    return best_threshold

def plot_confusion_matrix(y_true, y_scores, threshold):
    """
    Plots a confusion matrix using a specified threshold to binarize predictions.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_scores (array-like): Predicted similarity scores or probabilities.
        threshold (float): Threshold for classifying scores into binary labels.
    """
    y_pred = (y_scores > threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Spoof', 'Spoof'])

    plt.figure(figsize=(5, 4))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix at Threshold = {threshold:.3f}')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return roc_auc, fpr, tpr, thresholds

def find_best_threshold_accuracy(y_true, y_scores, thresholds):
    """
    Finds the best threshold that yields the highest accuracy.

    Args:
        y_true (array-like): Ground truth binary labels.
        y_scores (array-like): Predicted similarity scores or probabilities.
        thresholds (array-like): Threshold values to evaluate.

    Returns:
        float: Best accuracy
        float: Threshold that gives best accuracy
    """
    best_acc = 0
    best_acc_threshold = 0

    for t in thresholds:
        y_pred = (y_scores > t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_acc_threshold = t

    print(f"Best Accuracy: {best_acc:.4f} at Threshold: {best_acc_threshold:.3f}")
    return best_acc, best_acc_threshold

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer

# Load CLIP model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# Siamese Model using CLIP
class SiameseCLIPModel(nn.Module):
    def __init__(self, embedding_dim=512, projection_dim=128, freeze_clip=True):
        super(SiameseCLIPModel, self).__init__()
        self.clip = clip_model

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def encode(self, texts):
        inputs = clip_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.clip.device)
        with torch.no_grad():
            features = self.clip.get_text_features(**inputs)
        z = self.projector(features)
        return F.normalize(z, dim=1)

    def forward(self, text1, text2):
        z1 = self.encode(text1)
        z2 = self.encode(text2)
        return z1, z2

### HERE: TUNE MARGIN, higher margin = stronger separation!!!
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, z1, z2, label):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        cos_sim = F.cosine_similarity(z1, z2)
        cos_dist = 1 - cos_sim
        loss = label * cos_dist.pow(2) + (1 - label) * F.relu(self.margin - cos_dist).pow(2)
        return loss.mean()

from torch.utils.data import Dataset, DataLoader

class TextPairDataset(Dataset):
    def __init__(self, dataframe):
        self.name1 = dataframe['name1'].tolist()
        self.name2 = dataframe['name2'].tolist()
        self.label = dataframe['label'].tolist()

    def __len__(self):
        return len(self.name1)

    def __getitem__(self, idx):
        return self.name1[idx], self.name2[idx], torch.tensor(self.label[idx], dtype=torch.float32)