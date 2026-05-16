import os
import random
import shutil
from pathlib import Path

import cv2


DATASET_DIR = Path(r"E:\GIt\Mubser\dataset\obstacles_dataset")

CLASSES_FILE = DATASET_DIR / "_classes.txt"
ANNOTATIONS_FILE = DATASET_DIR / "_annotations.txt"

OUTPUT_DIR = Path(r"dataset\output")

USE_SUBSET = True
MAX_IMAGES = 3150   

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

RANDOM_SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}



def make_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def load_classes(classes_path: Path):
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def find_all_images(root: Path):
    images = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            images[p.name] = p
    return images


def parse_annotation_token(token: str):
    parts = token.split(",")
    if len(parts) < 5:
        return None

    try:
        x1 = float(parts[0])
        y1 = float(parts[1])
        x2 = float(parts[2])
        y2 = float(parts[3])
        class_id = int(parts[4])
        return x1, y1, x2, y2, class_id
    except ValueError:
        return None


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    box_w = x2 - x1
    box_h = y2 - y1
    x_center = x1 + box_w / 2.0
    y_center = y1 + box_h / 2.0

    return (
        x_center / img_w,
        y_center / img_h,
        box_w / img_w,
        box_h / img_h,
    )


def clamp01(v):
    return max(0.0, min(1.0, v))



def main():
    random.seed(RANDOM_SEED)

    if not CLASSES_FILE.exists():
        raise FileNotFoundError(f"Classes file not found: {CLASSES_FILE}")

    if not ANNOTATIONS_FILE.exists():
        raise FileNotFoundError(f"Annotations file not found: {ANNOTATIONS_FILE}")

    make_dirs()

    classes = load_classes(CLASSES_FILE)
    print(f"Loaded {len(classes)} classes")

    image_map = find_all_images(DATASET_DIR)
    print(f"Found {len(image_map)} image files in dataset folder")

    annotations_by_image = {}

    with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                print(f"Skipping bad line {line_num}: {line}")
                continue

            image_name = parts[0]
            box_tokens = parts[1:]

            parsed_boxes = []
            for token in box_tokens:
                parsed = parse_annotation_token(token)
                if parsed is not None:
                    parsed_boxes.append(parsed)

            if parsed_boxes:
                annotations_by_image[image_name] = parsed_boxes

    print(f"Found annotations for {len(annotations_by_image)} images")

    valid_images = [
        img_name for img_name in annotations_by_image
        if img_name in image_map
    ]

    print(f"Usable images: {len(valid_images)}")

    if USE_SUBSET:
        valid_images = valid_images[:MAX_IMAGES]
        print(f"Using subset of {len(valid_images)} images")

    random.shuffle(valid_images)

    n = len(valid_images)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_images = valid_images[:n_train]
    val_images = valid_images[n_train:n_train + n_val]
    test_images = valid_images[n_train + n_val:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    print("Split sizes:")
    for split_name, split_items in splits.items():
        print(f"  {split_name}: {len(split_items)}")

    for split_name, image_names in splits.items():
        for image_name in image_names:
            src_img_path = image_map[image_name]
            dst_img_path = OUTPUT_DIR / "images" / split_name / image_name

            shutil.copy2(src_img_path, dst_img_path)

            img = cv2.imread(str(src_img_path))
            if img is None:
                print(f"Could not read image: {src_img_path}")
                continue

            img_h, img_w = img.shape[:2]

            label_lines = []
            for (x1, y1, x2, y2, class_id) in annotations_by_image[image_name]:
                if class_id < 0 or class_id >= len(classes):
                    continue

                xc, yc, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)

                xc = clamp01(xc)
                yc = clamp01(yc)
                bw = clamp01(bw)
                bh = clamp01(bh)

                if bw <= 0 or bh <= 0:
                    continue

                label_lines.append(
                    f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
                )

            label_file = OUTPUT_DIR / "labels" / split_name / f"{Path(image_name).stem}.txt"
            with open(label_file, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {OUTPUT_DIR.resolve().as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for i, cls_name in enumerate(classes):
            f.write(f"  {i}: {cls_name}\n")

    print(f"\nDone. YOLO dataset created at: {OUTPUT_DIR}")
    print(f"data.yaml created at: {yaml_path}")


if __name__ == "__main__":
    main()