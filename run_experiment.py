"""
run_experiment.py — Entry Point Thực Nghiệm Fractal
=====================================================
Chạy script này để thực nghiệm phân tích FD & Lacunarity.

Chỉ cần chọn 2 thứ:
    1. Thư mục chứa TẤT CẢ ảnh + label (.txt) — bệnh và normal để chung
    2. Nơi lưu file CSV kết quả

Script tự phân loại:
    → .txt có nội dung = ảnh BỆNH  → crop ROI theo bounding box
    → .txt rỗng        = ảnh NORMAL → grid crop thành nhiều ô nhỏ

─── THAM SỐ THỰC NGHIỆM ───────────────────────────────────────────
Chỉnh các hằng số bên dưới để thử các cấu hình khác nhau,
không cần sửa gì thêm trong code.
────────────────────────────────────────────────────────────────────
"""

import os
import sys
import csv
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fractal.fractal_after_yolo import analyze_roi


# ════════════════════════════════════════════════════════════
#  THAM SỐ THỰC NGHIỆM — chỉnh tại đây, không cần đụng code
# ════════════════════════════════════════════════════════════

# Kích thước ROI tối thiểu (px) — bỏ qua ROI nhỏ hơn ngưỡng này.
# Lý do: fractal cần đủ scale để regression có ý nghĩa.
# Thử các giá trị: 32, 50, 64, 80
MIN_ROI_SIZE = 50

# Kích thước ô crop từ ảnh normal (px × px).
# Nên đặt gần với kích thước trung bình của ROI bệnh trong dataset.
# Thử các giá trị: 64, 96, 128, 160, 200
NORMAL_CROP_SIZE = 96

# Bước nhảy giữa các ô crop (px). Nhỏ hơn NORMAL_CROP_SIZE = có overlap.
# Overlap giúp lấy nhiều mẫu hơn, nhưng chạy lâu hơn.
# Thử: bằng NORMAL_CROP_SIZE (không overlap) hoặc NORMAL_CROP_SIZE // 2 (50% overlap)
NORMAL_CROP_STRIDE = 128

# Ngưỡng mean intensity để bỏ qua vùng đen (viền máy siêu âm, text).
# Crop có mean intensity < ngưỡng này sẽ bị bỏ qua.
# Thử các giá trị: 10, 20, 30
NORMAL_MIN_INTENSITY = 20

# Số lượng crop tối đa lấy từ mỗi ảnh normal (None = lấy tất cả).
# Giới hạn để tránh ảnh normal chiếm quá nhiều mẫu so với ảnh bệnh.
# Thử: None, 5, 10, 20
NORMAL_MAX_CROPS_PER_IMAGE = 10

# ════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────
# GUI HELPERS
# ─────────────────────────────────────────────

def select_folder(title):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder if folder else None


def select_save_file(title, default_name="fractal_experiment_results.csv"):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=default_name
    )
    root.destroy()
    return path if path else None


# ─────────────────────────────────────────────
# PARSE YOLO LABEL
# ─────────────────────────────────────────────

def parse_yolo_label_line(line, img_h, img_w):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bw       = float(parts[3])
        bh       = float(parts[4])
        conf     = float(parts[5]) if len(parts) > 5 else None
    except ValueError:
        return None

    x1 = max(0,     int((x_center - bw / 2) * img_w))
    y1 = max(0,     int((y_center - bh / 2) * img_h))
    x2 = min(img_w, int((x_center + bw / 2) * img_w))
    y2 = min(img_h, int((y_center + bh / 2) * img_h))
    return {'class_id': class_id, 'conf': conf,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}


# ─────────────────────────────────────────────
# GRID CROP CHO ẢNH NORMAL
# ─────────────────────────────────────────────

def grid_crop_normal(img, crop_size, stride, min_intensity, max_crops):
    """
    Chia ảnh normal thành các ô crop_size × crop_size với bước nhảy stride.
    Bỏ qua ô có mean intensity < min_intensity (vùng đen, viền máy).
    Trả về list các (crop, x1, y1, x2, y2).
    """
    h, w = img.shape[:2]
    crops = []

    for y in range(0, h - crop_size + 1, stride):
        for x in range(0, w - crop_size + 1, stride):
            patch = img[y:y + crop_size, x:x + crop_size]

            # Bỏ vùng đen
            if patch.mean() < min_intensity:
                continue

            crops.append((patch, x, y, x + crop_size, y + crop_size))

            if max_crops is not None and len(crops) >= max_crops:
                return crops

    return crops


# ─────────────────────────────────────────────
# XỬ LÝ TOÀN BỘ FOLDER
# ─────────────────────────────────────────────

def process_folder(data_dir):
    """
    Quét folder, tự phân loại disease/normal theo nội dung .txt,
    xử lý và trả về list kết quả.
    """
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = sorted([
        f for f in os.listdir(data_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])

    total = len(image_files)
    results = []
    count_disease_img  = 0
    count_normal_img   = 0
    count_disease_roi  = 0
    count_normal_crop  = 0
    count_skip_small   = 0
    count_skip_no_txt  = 0

    print(f"\n{'='*62}")
    print(f"  Thư mục  : {data_dir}")
    print(f"  Tổng ảnh : {total}")
    print(f"  Tham số  : MIN_ROI={MIN_ROI_SIZE}px | "
          f"CROP={NORMAL_CROP_SIZE}px | "
          f"STRIDE={NORMAL_CROP_STRIDE}px | "
          f"MAX_CROPS={NORMAL_MAX_CROPS_PER_IMAGE}")
    print(f"{'='*62}")

    for idx, img_name in enumerate(image_files, 1):
        base_name  = os.path.splitext(img_name)[0]
        img_path   = os.path.join(data_dir, img_name)
        label_path = os.path.join(data_dir, base_name + '.txt')

        prefix = f"  [{idx:>4}/{total}] {img_name}"

        # Không có .txt → bỏ qua
        if not os.path.exists(label_path):
            print(f"{prefix} → SKIP (không có .txt)")
            count_skip_no_txt += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"{prefix} → SKIP (không đọc được)")
            continue

        img_h, img_w = img.shape[:2]

        with open(label_path, 'r') as f:
            lines = [l for l in f.readlines() if l.strip()]

        # ── PHÂN LOẠI ──────────────────────────────
        if lines:
            # ── DISEASE: .txt có bounding box ──
            count_disease_img += 1
            img_has_valid_roi = False

            for line in lines:
                roi_info = parse_yolo_label_line(line, img_h, img_w)
                if roi_info is None:
                    continue

                x1, y1, x2, y2 = roi_info['x1'], roi_info['y1'], roi_info['x2'], roi_info['y2']
                roi_w = x2 - x1
                roi_h = y2 - y1
                roi   = img[y1:y2, x1:x2]

                # Lọc ROI quá nhỏ
                if roi.size == 0 or min(roi_w, roi_h) < MIN_ROI_SIZE:
                    note = f"ROI too small ({roi_w}×{roi_h}px < {MIN_ROI_SIZE}px)"
                    print(f"{prefix} [disease] → SKIP {note}")
                    results.append(_make_row(
                        img_name, 'disease',
                        x1, y1, x2, y2, roi_w, roi_h,
                        roi_info['conf'], np.nan, np.nan, note
                    ))
                    count_skip_small += 1
                    continue

                metrics = analyze_roi(roi)
                fd, lac = metrics['fd'], metrics['lacunarity']

                fd_str  = f"{fd:.4f}"  if not np.isnan(fd)  else "NaN"
                lac_str = f"{lac:.4f}" if not np.isnan(lac) else "NaN"
                print(f"{prefix} [disease] ROI {roi_w}×{roi_h} → FD={fd_str}  Lac={lac_str}")

                results.append(_make_row(
                    img_name, 'disease',
                    x1, y1, x2, y2, roi_w, roi_h,
                    roi_info['conf'], fd, lac, ''
                ))
                count_disease_roi += 1
                img_has_valid_roi = True

            if not img_has_valid_roi:
                print(f"{prefix} [disease] → Tất cả ROI bị lọc")

        else:
            # ── NORMAL: .txt rỗng → grid crop ──
            count_normal_img += 1
            crops = grid_crop_normal(
                img,
                crop_size=NORMAL_CROP_SIZE,
                stride=NORMAL_CROP_STRIDE,
                min_intensity=NORMAL_MIN_INTENSITY,
                max_crops=NORMAL_MAX_CROPS_PER_IMAGE
            )

            if not crops:
                print(f"{prefix} [normal]  → Không có crop hợp lệ (ảnh quá tối?)")
                continue

            print(f"{prefix} [normal]  → {len(crops)} crops")

            for patch, x1, y1, x2, y2 in crops:
                metrics = analyze_roi(patch)
                fd, lac = metrics['fd'], metrics['lacunarity']

                results.append(_make_row(
                    img_name, 'normal',
                    x1, y1, x2, y2,
                    NORMAL_CROP_SIZE, NORMAL_CROP_SIZE,
                    conf=None, fd=fd, lac=lac,
                    note=f'grid_crop_{NORMAL_CROP_SIZE}px'
                ))
                count_normal_crop += 1

    print(f"\n{'─'*62}")
    print(f"  Ảnh bệnh     : {count_disease_img} ảnh → {count_disease_roi} ROI hợp lệ")
    print(f"  ROI bị lọc   : {count_skip_small} (< {MIN_ROI_SIZE}px)")
    print(f"  Ảnh normal   : {count_normal_img} ảnh → {count_normal_crop} crops")
    print(f"  Bỏ qua       : {count_skip_no_txt} ảnh không có .txt")
    print(f"{'─'*62}")

    return results


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _make_row(img_name, label, x1, y1, x2, y2,
              roi_w, roi_h, conf, fd, lac, note=''):
    def fmt(v):
        if v is None:
            return 'N/A'
        if isinstance(v, float) and np.isnan(v):
            return 'NaN'
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    return {
        'image_name': img_name,
        'label':      label,
        'roi_x1':     x1,  'roi_y1': y1,
        'roi_x2':     x2,  'roi_y2': y2,
        'roi_width':  roi_w,
        'roi_height': roi_h,
        'yolo_conf':  fmt(conf),
        'fd':         fmt(fd),
        'lacunarity': fmt(lac),
        'note':       note,
    }


def save_csv(results, output_path):
    fieldnames = [
        'image_name', 'label',
        'roi_x1', 'roi_y1', 'roi_x2', 'roi_y2',
        'roi_width', 'roi_height',
        'yolo_conf', 'fd', 'lacunarity', 'note'
    ]
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[✓] Đã lưu {len(results)} dòng → {output_path}")


def print_summary(results):
    disease = [r for r in results if r['label'] == 'disease' and r['fd'] not in ('NaN','N/A')]
    normal  = [r for r in results if r['label'] == 'normal'  and r['fd'] not in ('NaN','N/A')]

    def stats(rows, col):
        vals = [float(r[col]) for r in rows if r[col] not in ('NaN', 'N/A')]
        if not vals:
            return "  (không có dữ liệu)"
        return (f"  n={len(vals)}"
                f"  mean={np.mean(vals):.4f}"
                f"  std={np.std(vals):.4f}"
                f"  min={np.min(vals):.4f}"
                f"  max={np.max(vals):.4f}"
                f"  median={np.median(vals):.4f}")

    print(f"\n{'='*62}")
    print("  TỔNG KẾT THỰC NGHIỆM")
    print(f"  Cấu hình: MIN_ROI={MIN_ROI_SIZE}px | CROP={NORMAL_CROP_SIZE}px | STRIDE={NORMAL_CROP_STRIDE}px")
    print(f"{'='*62}")
    print(f"[DISEASE] {len(disease)} ROI hợp lệ")
    print(f"  FD:        {stats(disease, 'fd')}")
    print(f"  Lacunarity:{stats(disease, 'lacunarity')}")
    print(f"\n[NORMAL]  {len(normal)} crops hợp lệ")
    print(f"  FD:        {stats(normal, 'fd')}")
    print(f"  Lacunarity:{stats(normal, 'lacunarity')}")
    print(f"{'='*62}")

    # Gợi ý về khả năng phân biệt
    d_fd  = [float(r['fd'])         for r in disease if r['fd']         not in ('NaN','N/A')]
    n_fd  = [float(r['fd'])         for r in normal  if r['fd']         not in ('NaN','N/A')]
    d_lac = [float(r['lacunarity']) for r in disease if r['lacunarity'] not in ('NaN','N/A')]
    n_lac = [float(r['lacunarity']) for r in normal  if r['lacunarity'] not in ('NaN','N/A')]

    if d_fd and n_fd:
        fd_sep = abs(np.mean(d_fd) - np.mean(n_fd)) / (np.std(d_fd) + np.std(n_fd) + 1e-9)
        print(f"\n  Separation index FD        = {fd_sep:.3f}  (cao hơn = phân biệt tốt hơn)")
    if d_lac and n_lac:
        lac_sep = abs(np.mean(d_lac) - np.mean(n_lac)) / (np.std(d_lac) + np.std(n_lac) + 1e-9)
        print(f"  Separation index Lacunarity = {lac_sep:.3f}  (cao hơn = phân biệt tốt hơn)")
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  FRACTAL EXPERIMENT — FD & Lacunarity Analysis")
    print("=" * 62)
    print(f"  Tham số hiện tại:")
    print(f"    MIN_ROI_SIZE            = {MIN_ROI_SIZE} px")
    print(f"    NORMAL_CROP_SIZE        = {NORMAL_CROP_SIZE} px")
    print(f"    NORMAL_CROP_STRIDE      = {NORMAL_CROP_STRIDE} px")
    print(f"    NORMAL_MIN_INTENSITY    = {NORMAL_MIN_INTENSITY}")
    print(f"    NORMAL_MAX_CROPS        = {NORMAL_MAX_CROPS_PER_IMAGE}")
    print()

    data_dir = select_folder("1/2 — Chọn thư mục chứa TẤT CẢ ảnh + label .txt")
    if not data_dir:
        print("Đã hủy."); return

    output_csv = select_save_file("2/2 — Chọn nơi lưu file CSV kết quả")
    if not output_csv:
        print("Đã hủy."); return

    results = process_folder(data_dir)

    if not results:
        print("\n[!] Không có kết quả. Kiểm tra lại thư mục đầu vào.")
        return

    save_csv(results, output_csv)
    print_summary(results)

    disease_n = sum(1 for r in results if r['label'] == 'disease' and r['fd'] not in ('NaN','N/A'))
    normal_n  = sum(1 for r in results if r['label'] == 'normal'  and r['fd'] not in ('NaN','N/A'))

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Hoàn thành!",
        f"Xử lý xong!\n\n"
        f"  Disease ROI hợp lệ: {disease_n}\n"
        f"  Normal crops      : {normal_n}\n\n"
        f"Kết quả lưu tại:\n{output_csv}"
    )
    root.destroy()


if __name__ == '__main__':
    main()
