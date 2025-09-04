"""
Augment per-class into a dedicated subfolder 'aug/' for each class.
Dataset layout (folder-per-class):
 dataset/
   crease/
     img001.jpg ...
   oil_spot/
     imgA.png ...

This script writes new images ONLY into dataset/<class>/aug/*.jpg
so you can clean or disable augmented data easily.

Target per class = max(original_count) by default, or --target.
Total after run per class ≈ max(original, target), counting aug files too.

Usage
-----
python scripts/augment_class_equalize_subfolder.py \
  --data ./dataset \
  --target 1200   # optional; default uses max original count
  --ext jpg       # output extension
  --per_class_cap 5000  # safety cap per class (aug + orig)
"""
import os, argparse, random
from typing import List, Dict
from collections import Counter
import numpy as np
import cv2

AUG_ROTATIONS = [-10, -5, 0, 5, 10]

# ------------------------ utils ------------------------
def is_img(name: str, exts=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
    return name.lower().endswith(exts)

def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for n in sorted(os.listdir(folder)):
        p = os.path.join(folder, n)
        if os.path.isfile(p) and is_img(n):
            out.append(p)
    return out

# ---------------------- augment ops --------------------
def aug_image(img):
    # geometry
    if random.random() < 0.5:
        img = cv2.flip(img, 1)  # horizontal
    if random.random() < 0.1:
        img = cv2.flip(img, 0)  # vertical
    ang = random.choice(AUG_ROTATIONS)
    if ang != 0:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # photometric
    alpha = 1.0 + random.uniform(-0.15, 0.25)  # contrast
    beta  = random.uniform(-15, 15)            # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    if random.random() < 0.3:
        noise = np.random.normal(0, 7, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img

# ------------------------ main -------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='Root folder that contains class folders')
    ap.add_argument('--target', type=int, default=0, help='Target images per class (orig + aug). 0 = use max original count')
    ap.add_argument('--ext', default='jpg', choices=['jpg','png','bmp'], help='Output image extension')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--per_class_cap', type=int, default=5000, help='Safety upper bound per class')
    args = ap.parse_args()

    random.seed(args.seed)

    classes = [d for d in sorted(os.listdir(args.data)) if os.path.isdir(os.path.join(args.data, d))]
    # Filter out a potential 'scripts' directory
    classes = [c for c in classes if c != 'scripts']
    if not classes:
        raise SystemExit('No class folders found under --data')

    # Count originals and existing aug
    orig_counts: Dict[str,int] = {}
    aug_counts: Dict[str,int]  = {}
    all_counts: Dict[str,int]  = {}

    for c in classes:
        c_dir = os.path.join(args.data, c)
        aug_dir = os.path.join(c_dir, 'aug')
        orig = list_images(c_dir)
        # exclude files inside aug/
        orig = [p for p in orig if os.path.dirname(p) != aug_dir]
        augs = list_images(aug_dir)
        orig_counts[c] = len(orig)
        aug_counts[c]  = len(augs)
        all_counts[c]  = orig_counts[c] + aug_counts[c]

    max_orig = max(orig_counts.values()) if orig_counts else 0
    target = args.target if args.target > 0 else max_orig

    print('[Classes]', classes)
    print('[Original counts]', orig_counts)
    print('[Existing aug counts]', aug_counts)
    print('[Current totals]', all_counts)
    print('[Target per class]', target)

    for c in classes:
        c_dir = os.path.join(args.data, c)
        aug_dir = os.path.join(c_dir, 'aug')
        os.makedirs(aug_dir, exist_ok=True)

        # Source pool = originals only (avoid chaining augment-of-augment)
        pool = [p for p in list_images(c_dir) if os.path.dirname(p) != aug_dir]
        if not pool:
            print(f'[WARN] No original images found for class {c}, skip')
            continue

        total = all_counts[c]
        idx = aug_counts[c]  # continue numbering
        cap = min(args.per_class_cap, max(target, total))

        while total < target and total < cap:
            src = random.choice(pool)
            img = cv2.imread(src)
            if img is None:
                continue
            aug = aug_image(img)
            name = f'aug_{idx:06d}.{args.ext}'
            out_path = os.path.join(aug_dir, name)
            # ensure unique name
            while os.path.exists(out_path):
                idx += 1
                name = f'aug_{idx:06d}.{args.ext}'
                out_path = os.path.join(aug_dir, name)
            # write
            if args.ext == 'jpg':
                cv2.imwrite(out_path, aug, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            else:
                cv2.imwrite(out_path, aug)
            idx += 1
            total += 1
        print(f'[DONE] {c}: orig={orig_counts[c]}, aug_added={max(0, total - (orig_counts[c] + aug_counts[c]))}, total={total}')

    print('All classes processed. You can include dataset/*/aug in your dataloader to use augmented data.')
