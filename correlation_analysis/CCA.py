import os
import numpy as np
import argparse
from collections import defaultdict
from sklearn.cross_decomposition import CCA
import random

def extract_id(filename):
    # Return the first 4 characters of the filename as ID
    return filename[:4]

def load_features_by_id(folder):
    # Load feature files from the folder and group them by ID
    d = defaultdict(list)
    for f in os.listdir(folder):
        if f.endswith('.npy'):
            d[extract_id(f)].append(np.load(os.path.join(folder, f)))
    return d

def collect_pairs(hr_by_id, lr_by_id, randomize=False, seed=0):
    hr_list, lr_list = [], []
    # Collect all paired features by ID
    for pid, hr_feats in hr_by_id.items():
        if pid not in lr_by_id:
            continue
        lr_feats = lr_by_id[pid]
        # Use the minimum number of samples available in both HR and LR for this ID
        n = min(len(hr_feats), len(lr_feats))
        for i in range(n):
            hr_list.append(hr_feats[i])
            lr_list.append(lr_feats[i])
    
    # Stack lists into numpy arrays
    hr_arr = np.stack(hr_list, axis=0)
    lr_arr = np.stack(lr_list, axis=0)
    
    # Random baseline: shuffle HR to break the one-to-one correspondence with LR
    if randomize:
        random.seed(seed)
        idx = list(range(len(hr_arr)))
        random.shuffle(idx)
        hr_arr = hr_arr[idx]
    
    return hr_arr, lr_arr

def main(hr_dir, lr_dir, n_components=3, randomize=False):
    # Load features grouped by ID for both HR and LR directories
    hr_by_id = load_features_by_id(hr_dir)
    lr_by_id = load_features_by_id(lr_dir)
    
    # Collect paired features (with optional randomization)
    X_hr, X_lr = collect_pairs(hr_by_id, lr_by_id, randomize=randomize)
    print(f"Loaded {X_hr.shape[0]} pairs, dim={X_hr.shape[1]}, randomize={randomize}")
    
    # Initialize and fit CCA model
    cca = CCA(n_components=n_components, max_iter=500)
    cca.fit(X_lr, X_hr)
    
    # Transform features to CCA space
    Xr, Yr = cca.transform(X_lr, X_hr)
    
    # Calculate correlation coefficients for each CCA component
    corr = []
    for i in range(n_components):
        # Compute covariance between corresponding CCA components
        cov = np.cov(Xr[:, i], Yr[:, i])[0, 1]
        # Compute total variance of the two components
        var = np.var(Xr[:, i]) + np.var(Yr[:, i])
        # Correlation coefficient (absolute value)
        corr.append(abs(cov) / var)
    
    print("Correlation coefficients:", [f"{c:.4f}" for c in corr])
    # Calculate squared correlation coefficients (r²)
    r2 = [c**2 for c in corr]
    print("r² values:", [f"{v:.4f}" for v in r2])
    print("Cumulative r²:", f"{sum(r2):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr", required=True, help="Directory containing high-resolution features")
    parser.add_argument("--lr", required=True, help="Directory containing low-resolution features")
    parser.add_argument("--n", type=int, default=3, help="Number of CCA components")
    parser.add_argument("--random", action="store_true",
                        help="Whether to use random baseline (shuffle pair relationships)")
    args = parser.parse_args()
    main(args.hr, args.lr, n_components=args.n, randomize=args.random)