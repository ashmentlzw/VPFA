import os
import shutil
import random
from collections import defaultdict

# Original image path (all images in one folder)
raw_dir = '/root/autodl-tmp/CAVIARa'  # All .jpg images
# Output root path
output_root = '/root/caviar_reset'

# Create output root directory
os.makedirs(output_root, exist_ok=True)

# Step 1: Collect all images
all_images = [f for f in os.listdir(raw_dir) if f.endswith('.jpg')]
pid_to_imgs = defaultdict(list)
for fname in all_images:
    pid = fname[:4]
    pid_to_imgs[pid].append(fname)

# Step 2: Filter out IDs with ≤10 images, keep only those with 20 images
valid_pid_map = {}
new_pid = 0
for pid, files in pid_to_imgs.items():
    if len(files) == 20:
        valid_pid_map[pid] = new_pid
        new_pid += 1

# Step 3: Preprocess all images (rename + save as intermediate set)
temp_dir = os.path.join(output_root, 'all_cleaned')
os.makedirs(temp_dir, exist_ok=True)

pid_to_cleaned = defaultdict(list)
for pid, new_pid in valid_pid_map.items():
    files = sorted(pid_to_imgs[pid])  
    for idx, fname in enumerate(files):
        cid = 0 if idx < 10 else 1
        index = fname[4:7] 
        new_name = f"{new_pid:04d}_c{cid+1}_{index}.jpg"
        shutil.copy(os.path.join(raw_dir, fname), os.path.join(temp_dir, new_name))
        pid_to_cleaned[new_pid].append(new_name)

# Step 4: Perform 10 splits
for split_id in range(10):
    split_dir = os.path.join(output_root, f"split{split_id}")
    train_dir = os.path.join(split_dir, 'bounding_box_train')
    query_dir = os.path.join(split_dir, 'query')
    gallery_dir = os.path.join(split_dir, 'bounding_box_test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(query_dir, exist_ok=True)
    os.makedirs(gallery_dir, exist_ok=True)

    all_pids = list(pid_to_cleaned.keys())
    random.seed(split_id)
    random.shuffle(all_pids)
    train_pids = all_pids[:25]
    test_pids = all_pids[25:]

    # Copy train images
    for pid in train_pids:
        for fname in pid_to_cleaned[pid]:
            shutil.copy(os.path.join(temp_dir, fname), os.path.join(train_dir, fname))

    # Query and gallery
    for pid in test_pids:
        hr_imgs = [f for f in pid_to_cleaned[pid] if '_c1_' in f]
        if not hr_imgs:
            print(f" Test ID {pid} has no HR images, cannot generate query")
            continue  # Skip this ID

        query_img = random.choice(hr_imgs)
        shutil.copy(os.path.join(temp_dir, query_img), os.path.join(query_dir, query_img))

        for fname in pid_to_cleaned[pid]:
            shutil.copy(os.path.join(temp_dir, fname), os.path.join(gallery_dir, fname))

