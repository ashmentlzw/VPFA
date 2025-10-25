import os
import shutil
import random
from PIL import Image
from pathlib import Path

def downsample_image(img: Image.Image, scale: int):
    w, h = img.size
    return img.resize((w // scale, h // scale), Image.BICUBIC)

def process_folder(src_folder: Path, dst_folder: Path):
    dst_folder.mkdir(parents=True, exist_ok=True)
    files = list(src_folder.glob("*.jpg"))

    pid_cam_map = {}
    for f in files:
        name = f.stem
        pid, camid, index = name.split("_")
        camid = int(camid[1])
        pid_cam_map.setdefault(pid, {}).setdefault(camid, []).append(f)

    for pid, cam_imgs in pid_cam_map.items():
        camids = list(cam_imgs.keys())
        lr_camid = random.choice(camids)
        scale = random.choice([2, 3, 4])

        for camid in camids:
            for img_path in cam_imgs[camid]:
                img = Image.open(img_path).convert("RGB")
                if camid == lr_camid:
                    # 下采样 + 重命名
                    down_img = downsample_image(img, scale)
                    new_name = img_path.stem + f"_{scale}x.jpg"
                    down_img.save(dst_folder / new_name)
                else:
                    # 高分辨率图像直接复制
                    shutil.copy(img_path, dst_folder / img_path.name)

def convert_all_splits(base_dir: Path):
    for i in range(10):
        print(f"Processing split{i} -> split{i}_mlr")
        src_split = base_dir / f"split{i}"
        dst_split = base_dir / f"split{i}_mlr"

        for subfolder in ["train", "query", "gallery"]:
            src = src_split / subfolder
            dst = dst_split / subfolder
            process_folder(src, dst)

if __name__ == "__main__":
    base_path = Path("/root/VIPeR_reset")
    convert_all_splits(base_path)
