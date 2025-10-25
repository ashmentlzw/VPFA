import os
import random
import shutil
from PIL import Image
from collections import defaultdict

# ====== Parameter Settings ======
DATA_ROOT   = "/mnt/bn/lzw-yg/MM/cuhk"      # Original CUHK03 dataset path
OUTPUT_ROOT = "/mnt/bn/lzw-yg/MM/MLR_CUHK03"  # Root directory for split results output
NUM_SPLITS  = 10                             # Number of splits
BASE_SEED   = 42                             # Base random seed, can be modified

# Downsampling ratios and corresponding suffixes
DOWNSAMPLE_OPTIONS = [
    (0.5,  "2x"),
    (1/3,  "3x"),
    (0.25, "4x"),
]

# Original subfolders
SUBDIRS = ["bounding_box_train", "bounding_box_test", "query"]

# ====== Step 1: Scan and parse all images ======
# Collection format: { id: [ { "cam": "c1"/"c2", "fname": filename, "src": subfolder } ] }
id_to_imgs = defaultdict(list)
for sub in SUBDIRS:
    folder = os.path.join(DATA_ROOT, sub)
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".png"):
            continue
        parts = fn.split("_")
        pid, cam = parts[0], parts[1]
        id_to_imgs[pid].append({
            "cam": cam,
            "fname": fn,
            "src": sub
        })

all_ids = sorted(id_to_imgs.keys())
num_ids = len(all_ids)
assert num_ids == 1467, f"Expected 1467 IDs, but scanned {num_ids}"

# ====== Step 2: Perform multiple splits in a loop ======
for split_idx in range(NUM_SPLITS, NUM_SPLITS+3):
    seed = BASE_SEED + split_idx
    random.seed(seed)
    # Randomly shuffle IDs, first 1367 for training, next 100 for testing
    ids = all_ids.copy()
    random.shuffle(ids)
    train_ids = set(ids[:1367])
    test_ids  = set(ids[1367:1367+100])

    # Create output directories
    split_dir = os.path.join(OUTPUT_ROOT, f"split_{split_idx}")
    train_dir = os.path.join(split_dir, "bounding_box_train")
    gallery_dir= os.path.join(split_dir, "bounding_box_test")
    query_dir = os.path.join(split_dir, "query")
    for d in (train_dir, gallery_dir, query_dir):
        os.makedirs(d, exist_ok=True)

    # ====== Step 3: Training set —— directly copy all images of training IDs ======
    for pid in train_ids:
        for info in id_to_imgs[pid]:
            src_path = os.path.join(DATA_ROOT, info["src"], info["fname"])
            dst_path = os.path.join(train_dir, info["fname"])
            shutil.copy(src_path, dst_path)

    # ====== Step 4: Test set —— build query (LR) and gallery (HR) ======
    for pid in test_ids:
        imgs = id_to_imgs[pid]
        # Classify by camera
        cam1 = [i for i in imgs if i["cam"] == "c1"]
        cam2 = [i for i in imgs if i["cam"] == "c2"]
        if not cam1 or not cam2:
            # Theoretically should not happen, all IDs have images from both sides
            continue

        # Randomly decide which side to use as LR probe
        if random.random() < 0.5:
            probe_imgs, gallery_imgs = cam1, cam2
        else:
            probe_imgs, gallery_imgs = cam2, cam1

        # （1）Process probe (all LR downsampling + add suffix)
        for info in probe_imgs:
            ratio, suf = random.choice(DOWNSAMPLE_OPTIONS)
            src_f = os.path.join(DATA_ROOT, info["src"], info["fname"])
            im = Image.open(src_f)
            new_size = (int(im.width * ratio), int(im.height * ratio))
            im2 = im.resize(new_size, Image.BILINEAR)

            name, ext = os.path.splitext(info["fname"])
            new_fn = f"{name}_{suf}{ext}"
            im2.save(os.path.join(query_dir, new_fn))

        # （2）Process gallery (randomly select 1 HR image)
        gal = random.choice(gallery_imgs)
        src_f = os.path.join(DATA_ROOT, gal["src"], gal["fname"])
        dst_f = os.path.join(gallery_dir, gal["fname"])
        shutil.copy(src_f, dst_f)

    print(f"Split {split_idx} completed, seed={seed}, train={len(train_ids)} IDs, test={len(test_ids)} IDs")