import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def pick_folder(title: str) -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    if not folder:
        print(f"[HUY] Khong chon thu muc: {title}")
        raise SystemExit(1)
    return Path(folder)


def save_image(path: Path, img: np.ndarray) -> bool:
    """Lưu ảnh an toàn, hỗ trợ đường dẫn Unicode."""
    ext = path.suffix.lower()
    success, buf = cv2.imencode(ext, img)
    if success:
        path.write_bytes(buf.tobytes())
        return True
    return False


def find_mask_for_image(image_path: Path, mask_folder: Path) -> Path | None:
    """Tìm file mask tương ứng với ảnh (cùng tên, khác đuôi)."""
    for ext in IMAGE_EXTENSIONS:
        candidate = mask_folder / (image_path.stem + ext)
        if candidate.exists():
            return candidate
    return None


def process_batch(
    image_folder: Path,
    mask_folder: Path,
    output_folder: Path,
    radius: int = 3,
    method: int = cv2.INPAINT_TELEA,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in image_folder.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        print("Khong tim thay file anh nao trong thu muc anh goc.")
        return

    success_count = 0
    skip_count    = 0

    for img_path in image_files:
        # --- Tìm mask tương ứng ---
        mask_path = find_mask_for_image(img_path, mask_folder)
        if mask_path is None:
            print(f"  [SKIP] Khong tim thay mask cho: {img_path.name}")
            skip_count += 1
            continue

        # --- Đọc ảnh gốc ---
        img_bytes = np.frombuffer(img_path.read_bytes(), dtype=np.uint8)
        original  = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        if original is None:
            print(f"  [SKIP] Khong doc duoc anh: {img_path.name}")
            skip_count += 1
            continue

        # --- Đọc mask ---
        mask_bytes = np.frombuffer(mask_path.read_bytes(), dtype=np.uint8)
        mask       = cv2.imdecode(mask_bytes, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  [SKIP] Khong doc duoc mask: {mask_path.name}")
            skip_count += 1
            continue

        # --- Kiểm tra kích thước ---
        if original.shape[:2] != mask.shape[:2]:
            print(f"  [SKIP] Kich thuoc khong khop: {img_path.name} vs {mask_path.name}")
            skip_count += 1
            continue

        # --- Inpaint ---
        result = cv2.inpaint(original, mask, radius, method)

        # --- Lưu kết quả ---
        out_name = img_path.stem + img_path.suffix.lower()
        out_path = output_folder / out_name
        if save_image(out_path, result):
            print(f"  [OK] {img_path.name} → {out_name}  ({original.shape[1]}x{original.shape[0]})")
            success_count += 1
        else:
            print(f"  [ERR] Luu that bai: {out_name}")
            skip_count += 1

    print(f"\n  Hoan tat: {success_count} anh da xu ly, {skip_count} file bi bo qua.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Fast Marching Inpaint - Batch Mode ===\n")
    print("Anh gốc => mask => output")

    image_folder  = pick_folder("Chon thu muc ANH GOC")
    mask_folder   = pick_folder("Chon thu muc MASK")
    output_folder = pick_folder("Chon thu muc OUTPUT (luu ket qua)")

    print(f"  Images : {image_folder}")
    print(f"  Masks  : {mask_folder}")
    print(f"  Output : {output_folder}")
    print(f"  Method : Fast Marching (TELEA), radius=3\n")

    process_batch(
        image_folder  = image_folder,
        mask_folder   = mask_folder,
        output_folder = output_folder,
        radius        = 3,
        method        = cv2.INPAINT_TELEA,
    )