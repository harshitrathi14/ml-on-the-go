"""
Feature-Tokenizer Transformer for tabular data (Gorishniy et al., 2021),
wrapped as a scikit-learn classifier.

Every input feature becomes a token (a learnt linear embedding of its
value), a [CLS] token is prepended, a small Transformer encoder attends
across the tokens and the [CLS] output is read out as the logit. Trained
with AdamW and early stopping on a validation AUC.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


def torch_available() -> bool:
    return torch is not None


def _device() -> "torch.device":
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if torch is not None:

    class _FTTransformer(nn.Module):
        def __init__(self, n_features: int, d_model: int, n_layers: int, n_heads: int, dropout: float):
            super().__init__()
            # One (weight, bias) pair per feature: value -> d_model token.
            self.weight = nn.Parameter(torch.empty(n_features, d_model))
            self.bias = nn.Parameter(torch.zeros(n_features, d_model))
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
                                               dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            tokens = x.unsqueeze(-1) * self.weight + self.bias           # (B, F, d)
            tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
            encoded = self.encoder(tokens)
            return self.head(self.norm(encoded[:, 0])).squeeze(-1)


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, d_model: int = 64, n_layers: int = 3, n_heads: int = 4, dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 1e-5, epochs: int = 40, batch_size: int = 512,
                 patience: int = 6, random_state: int = 42):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y) -> "FTTransformerClassifier":
        if torch is None:
            raise RuntimeError("PyTorch is not installed.")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        torch.manual_seed(self.random_state)
        device = _device()

        stratify = y if len(np.unique(y)) > 1 else None
        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.12, random_state=self.random_state, stratify=stratify)
        model = _FTTransformer(X.shape[1], self.d_model, self.n_layers, self.n_heads, self.dropout).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos_weight = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1.0)], device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.clamp(max=20.0))

        X_tr_t = torch.from_numpy(X_tr).to(device)
        y_tr_t = torch.from_numpy(y_tr).to(device)
        X_va_t = torch.from_numpy(X_va).to(device)
        best_auc, best_state, since_best = -1.0, None, 0
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        for epoch in range(self.epochs):
            model.train()
            perm = torch.randperm(len(y_tr), generator=generator).to(device)
            for start in range(0, len(perm), self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimiser.zero_grad(set_to_none=True)
                loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            auc = roc_auc_score(y_va, self._predict_logits(model, X_va_t)) if stratify is not None else 0.5
            if auc > best_auc + 1e-4:
                best_auc, since_best = auc, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                since_best += 1
                if since_best >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_ = model.eval()
        self.best_val_auc_ = float(best_auc)
        self.epochs_run_ = epoch + 1
        return self

    @staticmethod
    def _predict_logits(model, X_t: "torch.Tensor") -> np.ndarray:
        model.eval()
        outs = []
        with torch.no_grad():
            for start in range(0, len(X_t), 4096):
                outs.append(model(X_t[start:start + 4096]).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.empty(0)

    def predict_proba(self, X) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        device = next(self.model_.parameters()).device
        logits = self._predict_logits(self.model_, torch.from_numpy(X).to(device))
        p = 1 / (1 + np.exp(-logits))
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # The fitted network lives on the GPU; move it to CPU tensors for pickling
    # so the bundle loads on a machine (or process) without CUDA.
    def __getstate__(self):
        state = self.__dict__.copy()
        if "model_" in state:
            state["model_"] = state["model_"].to("cpu")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "model_" in state and torch is not None:
            self.model_ = self.model_.to(_device())
