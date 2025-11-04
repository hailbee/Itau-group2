import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

from model_utils.heads.pairwise_mlp import PairwiseMLP

class PairwiseFeatDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        self.x = torch.from_numpy(feats.astype(np.float32))
        self.y = torch.from_numpy(labels.astype(np.float32))
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def best_youden_threshold(y_true: np.ndarray, y_score: np.ndarray):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    youden = tpr - fpr
    i = int(np.argmax(youden))
    return float(thr[i]), float(tpr[i]), float(fpr[i]), float(youden[i])

def train_pairwise_mlp(
    train_feats: np.ndarray, train_labels: np.ndarray,
    val_feats: np.ndarray,   val_labels: np.ndarray,
    hidden=64, dropout=0.1, lr=1e-3, epochs=30, batch_size=256, weight_decay=1e-4,
    pos_weight: float | None = None, seed: int = 42, device: str = "cuda"
):
    torch.manual_seed(seed)
    model = PairwiseMLP(in_dim=train_feats.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    w = None if pos_weight is None else torch.tensor([pos_weight], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=w)

    train_dl = DataLoader(PairwiseFeatDataset(train_feats, train_labels), batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(PairwiseFeatDataset(val_feats,   val_labels),   batch_size=4096, shuffle=False)

    best_auc, best_sd = -1.0, None

    for _ in range(epochs):
        print(f"Epoch {_+1}/{epochs}")
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            all_logits, all_y = [], []
            for xb, yb in val_dl:
                xb = xb.to(device)
                all_logits.append(model(xb).cpu())
                all_y.append(yb)
            logits = torch.cat(all_logits).numpy()
            y_true = torch.cat(all_y).numpy()
            probs = 1 / (1 + np.exp(-logits))
            auc = roc_auc_score(y_true, probs)
            if auc > best_auc:
                best_auc = auc
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)
    return model, best_auc
