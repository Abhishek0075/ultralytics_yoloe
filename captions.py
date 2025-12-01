#!/usr/bin/env python3
"""
yolo_to_captions_v4.py

Generates simple captions for YOLO labels describing:
- Drone position (top/bottom + left/right/center)
- Drone size (tiny/small/medium/large)

Example output:
    "A small drone is in the left of the center of the image."
    "A large drone is in the top right of the image."

Usage:
    python yolo_to_captions_v4.py --images_dir images/ --labels_dir labels/ --out per_image
"""

import os
import glob
import argparse
from PIL import Image

def size_category(area):
    """Determine drone size based on pixel area."""
    if area < 100:
        return "tiny"
    elif area <= 1024:
        return "small"
    elif area <= 96 * 96:  # 9216 px
        return "medium"
    else:
        return "large"

def position_zone(cx, cy):
    """Determine the spatial zone from normalized center coordinates (0–1)."""
    if cx < 1/3 and cy < 1/3:
        return "top left"
    elif cx < 1/3 and cy > 2/3:
        return "bottom left"
    elif cx < 1/3:
        return "left of the center"
    elif cx > 2/3 and cy < 1/3:
        return "top right"
    elif cx > 2/3 and cy > 2/3:
        return "bottom right"
    elif cx > 2/3:
        return "right of the center"
    else:
        return "center"

def parse_yolo_line(line):
    """Parse one line of YOLO format: class cx cy w h"""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(float(parts[0]))
    cx, cy, w, h = map(float, parts[1:5])
    return cls, cx, cy, w, h

def process_image(image_path, labels_path, drone_class_id):
    """Generate captions for one image given YOLO labels."""
    img = Image.open(image_path)
    iw, ih = img.size
    captions = []

    if not os.path.exists(labels_path):
        return captions

    with open(labels_path, 'r') as f:
        lines = f.readlines()

    for ln in lines:
        parsed = parse_yolo_line(ln)
        if parsed is None:
            continue
        cls, cx, cy, w, h = parsed
        if drone_class_id >= 0 and cls != drone_class_id:
            continue

        bw, bh = w * iw, h * ih
        area = int(round(bw * bh))

        size = size_category(area)
        zone = position_zone(cx, cy)

        caption = f"A {size} drone is in the {zone} of the image."
        captions.append(caption)

    return captions

def main(images_dir, labels_dir, out_dir, out_mode, drone_class_id):
    """Main entry point for processing all images."""
    image_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images in {images_dir}")
    all_captions = []
    os.makedirs(out_dir, exist_ok=True)
    for img_path in image_files:
        print(f"Processing {img_path}...")
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + '.txt')
        captions = process_image(img_path, label_path, drone_class_id)

        if out_mode == 'per_image':
            print(f"{base}:")
            out_file = os.path.join(out_dir, base + '.caption.txt')
            with open(out_file, 'w') as f:
                if not captions:
                    f.write("No drone detected in this image.\n")
                else:
                    f.write("\n".join(captions) + "\n")
                    print(f"Saved captions to {out_file}")
        else:
            if not captions:
                all_captions.append(f"{base}: No drone detected.")
            else:
                for cap in captions:
                    all_captions.append(f"{base}: {cap}")

    if out_mode == 'single':
        out_file = os.path.join(out_dir, ".captions.txt")
        with open(out_file, 'w') as f:
            f.write("\n".join(all_captions))
        print(f"Saved all captions to {out_file}")
    else:
        print(f"Generated per-image caption files in {images_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', default='.', help='Directory containing images')
    parser.add_argument('--labels_dir', default=None, help='Directory containing YOLO .txt labels')
    parser.add_argument('--out_dir', help='Directory to save captions (default: same as images_dir)')
    parser.add_argument('--out', choices=['per_image','single'], default='per_image', help='Output mode')
    parser.add_argument('--drone_class', type=int, default=0, help='Class id for drone (use -1 for all classes)')
    args = parser.parse_args()

    labels_dir = args.labels_dir if args.labels_dir else args.images_dir
    main(args.images_dir, labels_dir, args.out_dir, args.out, args.drone_class)