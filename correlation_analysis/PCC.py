import os
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr

def extract_id(filename):
    # Extract the ID from the filename (first 4 characters)
    return filename[:4]

def load_features_by_id(folder):
    # Load feature files from the specified folder and group them by ID
    d = defaultdict(list)
    for f in os.listdir(folder):
        if f.endswith('.npy'):
            pid = extract_id(f)
            d[pid].append(np.load(os.path.join(folder, f)))
    return d

def compute_id_mean_diffs(hr_by_id, lr_by_id):
    """
    For each ID, compute the mean of all pairwise difference vectors (HR - LR)
    and record the number of sample pairs used.
    
    Returns three parallel lists:
      diffs: np.array with shape (num_ids, D) - mean difference vectors for each ID
      counts: list of int - number of sample pairs used for each ID
      ids: list of str - corresponding IDs
    """
    diffs = []
    counts = []
    ids = []
    for pid, hr_feats in hr_by_id.items():
        if pid not in lr_by_id:
            continue
        lr_feats = lr_by_id[pid]
        # Use the minimum number of samples available in both HR and LR for this ID
        n = min(len(hr_feats), len(lr_feats))
        if n == 0:
            continue
        # Calculate difference vectors for each sample pair
        id_diffs = [hr_feats[i] - lr_feats[i] for i in range(n)]
        # Store mean difference vector and sample count
        diffs.append(np.mean(id_diffs, axis=0))
        counts.append(n)
        ids.append(pid)
    return np.stack(diffs, axis=0), counts, ids

def main(hr_dir, lr_dir):
    # 1) Load features from high-resolution and low-resolution directories, grouped by ID
    hr_by_id = load_features_by_id(hr_dir)
    lr_by_id = load_features_by_id(lr_dir)

    # 2) Compute mean difference vectors for each ID and get the number of sample pairs per ID
    diffs, counts, ids = compute_id_mean_diffs(hr_by_id, lr_by_id)
    N, D = diffs.shape
    print(f"Total {N} IDs, feature dimension {D}")

    # 3) Calculate the global direction (mean of all ID mean difference vectors)
    global_dir = diffs.mean(axis=0)

    # 4) Compute Pearson correlation coefficient r between each ID's mean difference and global direction
    rs = []
    for i in range(N):
        r, _ = pearsonr(diffs[i], global_dir)
        rs.append(r)
    rs = np.array(rs)

    # 5) Group IDs by correlation threshold and calculate average sample counts for each group
    mask = rs > 0.4
    avg_count_high = np.mean([counts[i] for i in range(N) if mask[i]])
    avg_count_low  = np.mean([counts[i] for i in range(N) if not mask[i]])

    print(f"Number of IDs with Pearson r > 0.4: {mask.sum()}, average sample pairs: {avg_count_high:.2f}")
    print(f"Number of IDs with Pearson r ≤ 0.4: {N - mask.sum()}, average sample pairs: {avg_count_low:.2f}")

if __name__ == "__main__":
    main(
      hr_dir="/root/features/features_market_hr",
      lr_dir="/root/features/features_market_2x"
    )