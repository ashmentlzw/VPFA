import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time

# List of downsampling rates to train: modify here as needed
rates = [2, 3, 4]

# HR feature directory (keep unchanged)
hr_folder = "/root/features/features_viper7_hr"

# Output root directory
save_root = "/root/features/computed_features/simple_viper7"
os.makedirs(save_root, exist_ok=True)

# MLP definition remains unchanged
class MLP(nn.Module):
    def __init__(self, input_dim=3840, hidden_dim=2048):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Tanh()
        )

    def forward(self, x):
        return x + self.model(x)

# Compute segment-wise cosine similarity to monitor performance of each 768-dimensional block
def compute_cosine_similarity(X, Y):
    sims = []
    dim = 768
    for i in range(0, X.size(1), dim):
        sims.append(F.cosine_similarity(X[:, i:i+dim], Y[:, i:i+dim], dim=1).mean().item())
    return sims

# Train model for specified downsampling rate, using individual sample training instead of ID-based averaging
def train_for_rate(rate):
    print(f"\n=== Training MLP for {rate}x downsampling ===")
    lr_folder = f"/root/features/features_viper7_{rate}x"
    save_path = os.path.join(save_root, f"mlp_model_{rate}x.pth")

    # 1) Find common filenames between LR and HR
    files_lr = {f for f in os.listdir(lr_folder) if f.endswith(".npy")}
    files_hr = {f for f in os.listdir(hr_folder) if f.endswith(".npy")}
    common = sorted(files_lr & files_hr)

    # 2) Load files individually to construct X, Y
    feats_lr = []
    feats_hr = []
    for fname in common:
        lr_feat = np.load(os.path.join(lr_folder, fname))
        hr_feat = np.load(os.path.join(hr_folder, fname))
        feats_lr.append(lr_feat)
        feats_hr.append(hr_feat)

    # 3) Convert to Tensor and send to GPU
    X = torch.tensor(np.stack(feats_lr), dtype=torch.float32).cuda()
    Y = torch.tensor(np.stack(feats_hr), dtype=torch.float32).cuda()

    # 4) Initial cosine similarity
    init_mean = F.cosine_similarity(X, Y, dim=1).mean().item()
    init_splits = compute_cosine_similarity(X, Y)
    print(f"Initial mean cos: {init_mean:.6f}")
    for idx, v in enumerate(init_splits, 1):
        print(f"  Part {idx} cos: {v:.6f}")

    # 5) Create DataLoader
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 6) Model, optimizer, learning rate scheduler, loss function
    model = MLP(input_dim=X.size(1)).cuda()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_fn = nn.MSELoss()

    # 7) Training loop
    start_time = time.time()
    epochs = 120
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            pred = model(bx)
            loss = loss_fn(pred, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if epoch % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch:3d} loss {avg_loss:.6f}")
    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s")

    # 8) Post-training cosine similarity
    X_out = model(X)
    fin_mean = F.cosine_similarity(X_out, Y, dim=1).mean().item()
    fin_splits = compute_cosine_similarity(X_out, Y)
    print(f"Final mean cos: {fin_mean:.6f}")
    for idx, v in enumerate(fin_splits, 1):
        print(f"  Part {idx} cos: {v:.6f}")

    # 9) Save model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    for r in rates:
        train_for_rate(r)
    print("\nAll done.")
