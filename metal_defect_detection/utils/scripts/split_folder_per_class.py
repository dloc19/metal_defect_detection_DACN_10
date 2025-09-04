"""
Split a folder-per-class dataset into data/train, data/val, data/test.

Source layout (ví dụ hiện tại của bạn):
dataset/
  crease/
    img001.jpg
    ...
    aug/                 # optional, chứa ảnh augment
      aug_000001.jpg
  oil_spot/
  ... (10 classes)

Output layout (giống hình bạn muốn):
data/
  train/<class>/(originals train) + (aug/* nếu có)
  val/<class>/(chỉ original)
  test/<class>/(chỉ original)

Policy:
- val/test lấy ảnh gốc (không lấy từ aug/) → tránh leakage.
- train = phần còn lại của ảnh gốc + toàn bộ ảnh trong aug/.
- Mặc định COPY file. Có thể --move hoặc --symlink.

Usage:
python split_folder_per_class.py \
  --src ./dataset \
  --dst ./data \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --mode copy   # choices: copy|move|symlink
  --seed 42
"""
import os, argparse, random, shutil, sys

IMG_EXTS = ('.jpg','.jpeg','.png','.bmp','.tif','.tiff','.webp')

def is_img(p): 
    return p.lower().endswith(IMG_EXTS)

def list_images(folder):
    if not os.path.isdir(folder): return []
    return [os.path.join(folder,f) for f in sorted(os.listdir(folder)) if is_img(f)]

def list_aug_images(class_dir):
    aug_dir = os.path.join(class_dir, 'aug')
    if not os.path.isdir(aug_dir): return []
    return [os.path.join(aug_dir,f) for f in sorted(os.listdir(aug_dir)) if is_img(f)]

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def place(src, dst, mode='copy'):
    ensure_dir(os.path.dirname(dst))
    if mode == 'copy':
        shutil.copy2(src, dst)
    elif mode == 'move':
        shutil.move(src, dst)
    elif mode == 'symlink':
        try:
            if os.path.lexists(dst): os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
        except Exception:
            shutil.copy2(src, dst)
    else:
        raise ValueError('mode must be copy|move|symlink')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Path to source dataset folder-per-class')
    ap.add_argument('--dst', required=True, help='Path to output data folder (will create train/val/test)')
    ap.add_argument('--val_ratio', type=float, default=0.1)
    ap.add_argument('--test_ratio', type=float, default=0.1)
    ap.add_argument('--mode', choices=['copy','move','symlink'], default='copy')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    classes = [d for d in sorted(os.listdir(args.src)) 
               if os.path.isdir(os.path.join(args.src,d)) and d != 'scripts']
    if not classes:
        print("No class folders found under", args.src, file=sys.stderr)
        sys.exit(1)

    # Prepare output dirs
    for split in ['train','val','test']:
        for c in classes:
            ensure_dir(os.path.join(args.dst, split, c))

    for c in classes:
        cdir = os.path.join(args.src, c)
        aug = list_aug_images(cdir)                # all aug images
        orig = list_images(cdir)                   # include all files
        # filter out any file that sits inside aug/
        orig = [p for p in orig if os.path.dirname(p) != os.path.join(cdir, 'aug')]

        n = len(orig)
        nv = int(round(n * args.val_ratio))
        nt = int(round(n * args.test_ratio))
        if nv + nt > n:
            nt = max(0, n - nv)

        random.shuffle(orig)
        val_orig  = orig[:nv]
        test_orig = orig[nv:nv+nt]
        train_orig= orig[nv+nt:]

        # Copy/Move/Symlink
        for p in train_orig:
            dst = os.path.join(args.dst, 'train', c, os.path.basename(p))
            place(p, dst, args.mode)
        for p in val_orig:
            dst = os.path.join(args.dst, 'val', c, os.path.basename(p))
            place(p, dst, args.mode)
        for p in test_orig:
            dst = os.path.join(args.dst, 'test', c, os.path.basename(p))
            place(p, dst, args.mode)

        # Add aug images to train only
        for p in aug:
            dst = os.path.join(args.dst, 'train', c, 'aug_' + os.path.basename(p))
            place(p, dst, args.mode)

        print(f"[{c}] orig={n} -> train={len(train_orig)} (+aug={len(aug)}), "
              f"val={len(val_orig)}, test={len(test_orig)}")

    print("Done. Output at", args.dst)
    print("Structure: data/{train,val,test}/<class>")

if __name__ == '__main__':
    main()
