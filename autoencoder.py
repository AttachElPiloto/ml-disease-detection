import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import fbeta_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


class AE(nn.Module):
    def __init__(self, in_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z


def train_autoencoder(
    Xtr_z,
    latent_dim=64,
    max_epochs=200,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-5,
    patience=10,
    device="cpu",
):
    Xtr = Xtr_z.astype(np.float32)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)

    train_ds = TensorDataset(Xtr_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    ae = AE(Xtr.shape[1], latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()

    best_loss, stall = float("inf"), 0
    best_state = None

    for epoch in range(max_epochs):
        ae.train()
        epoch_loss = 0.0

        for (xb,) in train_dl:
            xrec, _ = ae(xb)
            loss = crit(xrec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)

        epoch_loss /= len(train_ds)

        if epoch_loss + 1e-6 < best_loss:
            best_loss = epoch_loss
            stall = 0
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
        else:
            stall += 1

        if stall >= patience:
            break

    ae.load_state_dict(best_state)
    ae.to(device).eval()
    print(f"Autoencoder trained; best reconstruction loss: {best_loss:.4f}")
    return ae


@torch.no_grad()
def encode(ae, X, device="cpu"):
    """
    Encode data X into latent space using a trained autoencoder.
    """
    X_t = torch.tensor(X.astype(np.float32), dtype=torch.float32, device=device)
    _, Z = ae(X_t)
    return Z.cpu().numpy()


from sklearn.model_selection import train_test_split

def run_xgb_on_latent(
    Xtr_z,
    Xte_z,
    y_tr,
    y_te,
    latent_dim=64,
    beta=2.0,
    device="cpu",
    plot=True,
):
    """
    1) Train AE on Xtr_z
    2) Encode train/test into latent space
    3) Train XGBoost on latent features with a validation set + early stopping
    """
    # 1 Train AE and encode
    ae = train_autoencoder(Xtr_z, latent_dim=latent_dim, device=device)
    Z_tr_full = encode(ae, Xtr_z, device=device)
    Z_te      = encode(ae, Xte_z, device=device)

    print("Encoded train shape:", Z_tr_full.shape)

    # 2  Prepare labels
    le = LabelEncoder()
    y_tr_enc_full = le.fit_transform(y_tr)
    y_te_enc      = le.transform(y_te)

    # 3 Train/val split in latent space
    Z_tr, Z_val, y_tr_enc, y_val_enc = train_test_split(
        Z_tr_full,
        y_tr_enc_full,
        test_size=0.2,
        random_state=42,
        stratify=y_tr_enc_full,
    )

    # 4 XGBoost on latent space
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        tree_method="hist",
        n_estimators=6000,
        learning_rate=0.3,
        subsample=0.4,
        colsample_bytree=0.35,
        reg_lambda=1.0,
        reg_alpha=0,
        random_state=42,
    )

    xgb.fit(
        Z_tr_full,
        y_tr_enc_full,
        verbose=False,
    )

    # 5 Predictions
    y_tr_pred = xgb.predict(Z_tr)
    y_val_pred = xgb.predict(Z_val)
    y_te_pred = xgb.predict(Z_te)

    fb_train = fbeta_score(y_tr_enc,  y_tr_pred,  average="macro", beta=beta)
    fb_val   = fbeta_score(y_val_enc, y_val_pred, average="macro", beta=beta)
    fb_test  = fbeta_score(y_te_enc,  y_te_pred,  average="macro", beta=beta)

    print("=== XGBoost on AE latent space (with validation) ===")
    print(f"Train Fβ={beta} (macro): {fb_train:.3f}")
    print(f"Val   Fβ={beta} (macro): {fb_val:.3f}")
    print(f"Test  Fβ={beta} (macro): {fb_test:.3f}")

    if plot:
        cm_test = confusion_matrix(le.inverse_transform(y_te_enc),
                                   le.inverse_transform(y_te_pred),
                                   labels=le.classes_)
        ConfusionMatrixDisplay(cm_test, display_labels=le.classes_).plot(
            cmap="Blues", colorbar=False
        )
        plt.title("AE + XGB – Test")
        plt.tight_layout()
        plt.show()

    metrics = {
        "name": f"AE(latent={latent_dim})+XGB",
        "features": Z_tr_full.shape[1],
        "var_quantile": 0.96,
        "Fbeta_train": float(fb_train),
        "Fbeta_val": float(fb_val),
        "Fbeta_test": float(fb_test),
    }

    return ae, xgb, le, metrics


import pickle
import torch

def save_latent_pipeline(
    save_dir,
    ae,
    xgb,
    le,
    mask,
    scaler,
    gene_order,
    latent_dim
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Autoencoder
    torch.save(ae.state_dict(), f"{save_dir}/ae.pt")

    # 2. XGBoost
    xgb.save_model(f"{save_dir}/xgb.json")

    # 3. Other objects
    obj = {
        "label_encoder": le,
        "mask": mask,
        "scaler": scaler,
        "gene_order": gene_order,
        "latent_dim": latent_dim,
    }
    with open(f"{save_dir}/pipeline.pkl", "wb") as f:
        pickle.dump(obj, f)

    print(f"Pipeline saved to {save_dir}")


def load_latent_pipeline(save_dir, device="cpu"):
    with open(f"{save_dir}/pipeline.pkl", "rb") as f:
        meta = pickle.load(f)

    latent_dim = meta["latent_dim"]
    gene_order = meta["gene_order"]

    # Build autoencoder again
    ae = AE(in_dim=977, latent_dim=latent_dim).to(device)
    ae.load_state_dict(torch.load(f"{save_dir}/ae.pt", map_location=device))
    ae.eval()

    # Load XGBoost
    xgb = XGBClassifier()
    xgb.load_model(f"{save_dir}/xgb.json")

    return ae, xgb, meta


