import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import re
from config import cfg
from datasets import make_dataloader
from model import make_model

# -----------------------------------------------------------------------------
# User static configuration area (set according to your needs)
# -----------------------------------------------------------------------------
CONFIG_FILE = "/mnt/bn/lzw-yg/MM/TransReID-main/configs/Market/vit_transreid_stride0.yml"
MODEL_WEIGHT_PATH = "/mnt/bn/lzw-yg/MM/TransReID-main/vit_transreid_market.pth"

INPUT_ROOT = "/mnt/bn/lzw-yg/MM/sr_market"
OUTPUT_ROOT = "/mnt/bn/lzw-yg/MM/sr_market_feats"

BATCH_SIZE = 64
NUM_WORKERS = 5
DEVICE = 'cuda'

SPLIT_COUNT = 1
INPUT_SUBDIRS = [
    "bounding_box_train",
    "bounding_box_train_2x",
    "bounding_box_train_3x",
    "bounding_box_train_4x"
]
# -----------------------------------------------------------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        fn = os.path.basename(path)
        # Use regular expression to extract person ID and camera ID
        match = re.match(r'(\d+)_c(\d+)_.*', fn)
        if match:
            pid = int(match.group(1))
            cam = int(match.group(2)) - 1  # Camera ID starts from 0
        else:
            # If filename format does not match, use default values
            pid = -1
            cam = -1
        view = cam
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Failed to open image {path}: {e}")
            raise
        img = self.transform(img)
        return img, pid, cam, view, path

def main():
    # Load configuration
    cfg.merge_from_file(CONFIG_FILE)
    cfg.defrost()
    if MODEL_WEIGHT_PATH:
        cfg.TEST.WEIGHT = MODEL_WEIGHT_PATH

    cfg.DATASETS.ROOT_DIR = os.path.join(INPUT_ROOT, "split_1")
    cfg.freeze()

    # Get model required information
    _, _, _, _, num_classes, camera_num, view_num = make_dataloader(cfg)

    # Build model
    model = make_model(cfg,
                       num_class=751,
                       camera_num=camera_num,
                       view_num=view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.to(DEVICE).eval()

    # Preprocessing
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # Iterate through split_1
    for split_idx in range(SPLIT_COUNT, SPLIT_COUNT + 1):
        for sub in INPUT_SUBDIRS:
            inp_dir = os.path.join(INPUT_ROOT, f"split_{split_idx}", "market1501", sub)
            if not os.path.isdir(inp_dir):
                print(f"Skipping non-existent directory: {inp_dir}")
                continue

            img_list = [
                os.path.join(inp_dir, fn)
                for fn in os.listdir(inp_dir)
                if fn.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
            if not img_list:
                continue

            ds = ImageFolderDataset(img_list, val_transforms)
            loader = DataLoader(ds,
                                batch_size=BATCH_SIZE,
                                shuffle=False,
                                num_workers=NUM_WORKERS,
                                pin_memory=True)

            out_dir = os.path.join(OUTPUT_ROOT, f"split_{split_idx}", sub)
            os.makedirs(out_dir, exist_ok=True)

            # Inference
            with torch.no_grad():
                for imgs, pids, cams, views, paths in loader:
                    imgs = imgs.to(DEVICE)
                    pids = torch.tensor(pids).to(DEVICE)
                    cams = torch.tensor(cams).to(DEVICE)
                    views = torch.tensor(views).to(DEVICE)

                    # Forward inference
                    feat1 = model(imgs, cam_label=cams, view_label=views)
                    feat1 = feat1[1] if isinstance(feat1, (tuple, list)) else feat1
                    imgs_flip = torch.flip(imgs, dims=[3])
                    feat2 = model(imgs_flip, cam_label=cams, view_label=views)
                    feat2 = feat2[1] if isinstance(feat2, (tuple, list)) else feat2

                    # Average features
                    feat = (feat1 + feat2) * 0.5
                    feat = feat.cpu().numpy()

                    # Save features
                    for vec, path in zip(feat, paths):
                        name = os.path.splitext(os.path.basename(path))[0]
                        np.save(os.path.join(out_dir, name + ".npy"), vec)

            print(f"[split_{split_idx} | {sub}] Inference completed, saved to {out_dir}")

if __name__ == "__main__":
    main()
