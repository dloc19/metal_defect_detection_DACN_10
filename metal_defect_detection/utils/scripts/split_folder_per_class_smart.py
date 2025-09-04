"""
Smart splitter for folder-per-class datasets with class-wise minimums.

Fixes the issue where small classes end up with only 5–8 images in val/test.

Policy
------
- Start from desired ratios (e.g. 70/15/15 via --val_ratio 0.15 --test_ratio 0.15).
- Enforce --min_val and --min_test per class (on ORIGINAL images only).
- If a class is too small to satisfy both minimums:
  * If --collapse_test_when_small is set (default behavior when passed),
    move test into val (test=0).
  * Otherwise, reduce test to fit what's available (ensuring non-negative).
- Train = remaining originals + ALL images under <class>/aug/* (if exists).
- val/test NEVER contain aug images.

Usage (PowerShell one-liner)
----------------------------
python .\\dataset\\scripts\\split_folder_per_class_smart.py ^
  --src ".\\dataset" ^
  --dst ".\\data" ^
  --val_ratio 0.15 ^
  --test_ratio 0.15 ^
  --min_val 10 ^
  --min_test 10 ^
  --collapse_test_when_small ^
  --mode copy ^
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

def allocate_counts(n, val_ratio, test_ratio, min_val, min_test, collapse_test_when_small=True):
    """
    Compute numbers (nv, nt, ntrain) from n originals.
    Enforce minimums; adjust gracefully for small classes.
    """
    # Ratio-based initial targets
    nv = int(round(n * val_ratio))
    nt = int(round(n * test_ratio))

    # Enforce minimum for val
    if n >= min_val:
        nv = max(nv, min_val)
    else:
        nv = min(n, max(0, min_val if min_val <= n else n))

    # Keep at least 1 for train when possible
    if nv >= n:
        nv = max(0, n - 1)

    # Remaining for test + train
    remain = n - nv

    # Enforce minimum for test from the remainder
    if remain >= min_test:
        nt = max(nt, min_test)
    else:
        nt = min(remain, min_test)

    # If nv + nt > n, reduce test first, then val
    if nv + nt > n:
        excess = nv + nt - n
        reduc = min(excess, nt)
        nt -= reduc
        excess -= reduc
        if excess > 0:
            nv = max(0, nv - excess)

    # Optionally collapse tiny test into val
    if collapse_test_when_small and (n < (min_val + min_test + 2) or nt == 0):
        nt = 0
        # Try to keep a healthy val (but never exceed n-1 to leave something for train)
        nv = min(n - 1 if n > 1 else n, max(nv, min_val if n >= min_val else nv))

    # Final clamps, ensure non-negative and leave at least 1 for train if possible
    nv = max(0, min(nv, n))
    nt = max(0, min(nt, n - nv))
    ntrain = max(0, n - nv - nt)
    if n == 1:
        nv, nt, ntrain = 0, 0, 1

    return nv, nt, ntrain

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Path to source dataset folder-per-class')
    ap.add_argument('--dst', required=True, help='Path to output data folder (will create train/val/test)')
    ap.add_argument('--val_ratio', type=float, default=0.15)
    ap.add_argument('--test_ratio', type=float, default=0.15)
    ap.add_argument('--min_val', type=int, default=10, help='Minimum ORIGINAL images per class in val (if available)')
    ap.add_argument('--min_test', type=int, default=10, help='Minimum ORIGINAL images per class in test (if available)')
    ap.add_argument('--collapse_test_when_small', action='store_true', help='If set, drop test -> val for tiny classes')
    ap.add_argument('--mode', choices=['copy','move','symlink'], default='copy')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    classes = [d for d in sorted(os.listdir(args.src)) 
               if os.path.isdir(os.path.join(args.src,d)) and d.lower() != 'scripts']
    if not classes:
        print("No class folders found under", args.src, file=sys.stderr); sys.exit(1)

    # Prepare output dirs
    for split in ['train','val','test']:
        for c in classes:
            ensure_dir(os.path.join(args.dst, split, c))

    for c in classes:
        cdir = os.path.join(args.src, c)
        aug = list_aug_images(cdir)                # all aug images
        orig = list_images(cdir)                   # includes all files
        # filter out any file inside aug/
        orig = [p for p in orig if os.path.dirname(p) != os.path.join(cdir, 'aug')]
        n = len(orig)

        nv, nt, ntrain = allocate_counts(
            n, args.val_ratio, args.test_ratio,
            args.min_val, args.min_test,
            args.collapse_test_when_small
        )

        random.shuffle(orig)
        val_orig  = orig[:nv]
        test_orig = orig[nv:nv+nt]
        train_orig= orig[nv+nt:]  # remaining originals

        # Place files
        for p in train_orig:
            dst = os.path.join(args.dst, 'train', c, os.path.basename(p))
            place(p, dst, args.mode)
        for p in val_orig:
            dst = os.path.join(args.dst, 'val', c, os.path.basename(p))
            place(p, dst, args.mode)
        for p in test_orig:
            dst = os.path.join(args.dst, 'test', c, os.path.basename(p))
            place(p, dst, args.mode)

        # Put aug images ONLY into train split
        for p in aug:
            dst = os.path.join(args.dst, 'train', c, 'aug_' + os.path.basename(p))
            place(p, dst, args.mode)

        print(f"[{c}] orig={n} -> train={len(train_orig)} (+aug={len(aug)}), val={len(val_orig)}, test={len(test_orig)}")

    print("Done. Output at", args.dst)
    print("Structure: data/{train,val,test}/<class>")

if __name__ == '__main__':
    main()
