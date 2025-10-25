import os
import random
from PIL import Image
from collections import defaultdict

random.seed(42)

input_dir = '/mnt/bn/lzw-yg/MM/TransReID-main/market1501'   # Original high-resolution dataset root directory
output_dir = '/mnt/bn/lzw-yg/MM/TransReID-main/mlr_market1501' # Output directory with 3 scales after expansion

# Directly change the list to 6 downsample rates
downsample_rates = [2, 3, 4]

def downsample_image(img, rate):
    new_size = (img.width // rate, img.height // rate)
    return img.resize(new_size, Image.LANCZOS)

stats = defaultdict(int)

for folder in ['bounding_box_train', 'bounding_box_test', 'query']:
    in_folder = os.path.join(input_dir, folder)
    out_folder = os.path.join(output_dir, folder)
    os.makedirs(out_folder, exist_ok=True)

    # All person IDs
    pids = set(f.split('_')[0]
               for f in os.listdir(in_folder)
               if f.endswith('.jpg') and '_' in f)

    for pid in pids:
        # All images under this ID
        imgs = [f for f in os.listdir(in_folder)
                if f.startswith(pid + '_') and f.endswith('.jpg')]
        # If there are multiple cameras, randomly select one cam for downsampling
        cams = {f.split('_')[1] for f in imgs}
        chosen_cam = random.choice(list(cams)) if len(cams) > 1 else None

        for name in imgs:
            basename, ext = name.rsplit('.', 1)
            cam = basename.split('_')[1]
            src = os.path.join(in_folder, name)
            try:
                img = Image.open(src)
            except:
                continue

            # If this camera is selected, randomly choose a rate
            if chosen_cam and cam == chosen_cam:
                rate = random.choice(downsample_rates)
                lr = downsample_image(img, rate)
                new_name = f"{basename}_{rate}x.{ext}"
                stats[f"{rate}x"] += 1
                out_img = lr
            else:
                new_name = name
                stats['HR'] += 1
                out_img = img

            dst = os.path.join(out_folder, new_name)
            out_img.save(dst)
