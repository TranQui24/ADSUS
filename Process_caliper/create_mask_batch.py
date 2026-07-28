import cv2
import numpy as np
import os
import argparse
from pathlib import Path


def yolo_to_binary_mask(yolo_lines, img_width, img_height, target_class=0):
    """
    Chuyển đổi các dòng YOLO format sang binary mask.
    Chỉ xử lý các bbox có class == target_class.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        if class_id != target_class:
            continue

        x_center   = float(parts[1])
        y_center   = float(parts[2])
        box_width  = float(parts[3])
        box_height = float(parts[4])

        x_min = int((x_center - box_width  / 2) * img_width)
        y_min = int((y_center - box_height / 2) * img_height)
        x_max = int((x_center + box_width  / 2) * img_width)
        y_max = int((y_center + box_height / 2) * img_height)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(img_width,  x_max)
        y_max = min(img_height, y_max)

        mask[y_min:y_max, x_min:x_max] = 255

    return mask


def find_image_for_label(label_path: Path, image_folder: Path):
    """
    Tìm file ảnh gốc tương ứng với file label (cùng tên, khác đuôi).
    Hỗ trợ nhiều định dạng ảnh phổ biến.
    """
    stem = label_path.stem
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    for ext in image_extensions:
        candidate = image_folder / (stem + ext)
        if candidate.exists():
            return candidate

    return None


def process_batch(
    label_folder: str,
    image_folder: str,
    output_folder: str,
    target_class: int = 0,
    output_ext: str = ".png",
):
    label_dir  = Path(label_folder)
    image_dir  = Path(image_folder)
    output_dir = Path(output_folder)

    if not label_dir.exists():
        raise FileNotFoundError(f"Thư mục label không tồn tại: {label_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    label_files = sorted(label_dir.glob("*.txt"))

    if not label_files:
        print("⚠  Không tìm thấy file .txt nào trong thư mục label.")
        return

    success_count = 0
    skip_count    = 0

    for label_path in label_files:
        # --- Tìm ảnh gốc ---
        image_path = find_image_for_label(label_path, image_dir)
        if image_path is None:
            print(f"  [SKIP] Không tìm thấy ảnh cho: {label_path.name}")
            skip_count += 1
            continue

        # --- Đọc kích thước ảnh ---
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  [SKIP] Không đọc được ảnh: {image_path.name}")
            skip_count += 1
            continue

        img_height, img_width = img.shape[:2]

        # --- Đọc file label ---
        with open(label_path, "r", encoding="utf-8") as f:
            yolo_lines = f.readlines()

        # --- Tạo mask ---
        mask = yolo_to_binary_mask(yolo_lines, img_width, img_height, target_class)

        # --- Lưu mask (dùng imencode để tránh lỗi Unicode trong đường dẫn) ---
        output_filename = image_path.stem + output_ext
        output_path = output_dir / output_filename
        success, buf = cv2.imencode(output_ext, mask)
        if success:
            output_path.write_bytes(buf.tobytes())
        else:
            print(f"  [ERR] Khong encode duoc mask: {output_filename}")
            skip_count += 1
            continue

        print(f"  [OK] {label_path.name} → {output_filename}  ({img_width}x{img_height})")
        success_count += 1

    print(f"\n  Hoan tat: {success_count} mask da tao, {skip_count} file bi bo qua.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def pick_folder(title: str) -> str:
    """Mở hộp thoại chọn thư mục, trả về đường dẫn hoặc thoát nếu huỷ."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    folder = filedialog.askdirectory(title=title)
    root.destroy()

    if not folder:
        print(f"[HUY] Nguoi dung khong chon thu muc: {title}")
        raise SystemExit(1)
    return folder


if __name__ == "__main__":
    print("Label => ảnh gốc => output")
    label_folder  = pick_folder("Chon thu muc LABEL (.txt YOLO)")
    image_folder  = pick_folder("Chon thu muc ANH GOC")
    output_folder = pick_folder("Chon thu muc OUTPUT (luu mask)")

    print(f"  Labels : {label_folder}")
    print(f"  Images : {image_folder}")
    print(f"  Output : {output_folder}\n")

    process_batch(label_folder, image_folder, output_folder, target_class=0, output_ext=".png")
