import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Đọc ảnh từ đường dẫn có ký tự Unicode (tiếng Việt, v.v.).
    cv2.imread trên Windows không hỗ trợ Unicode, dùng np.fromfile + imdecode thay thế."""
    path = os.path.normpath(path)
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            print(f"  [LỖI] File rỗng hoặc không tồn tại: '{path}'")
            return None
        img = cv2.imdecode(buf, flags)
        if img is None:
            print(f"  [LỖI] cv2.imdecode thất bại (file hỏng hoặc sai định dạng): '{path}'")
        return img
    except Exception as e:
        print(f"  [LỖI] imread_unicode thất bại cho '{path}': {e}")
        return None

def pick_folder(title):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title)
        root.destroy()
        if not path:
            print("Bạn chưa chọn thư mục, thử lại.")
            return pick_folder(title)
        return path
    except Exception as e:
        print(f"(Không mở được hộp thoại chọn thư mục: {e})")
        return input(f"{title} - dán đường dẫn thư mục vào đây: ").strip()


def pick_save_file(title, default_name="ket_qua_inpaint.csv"):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.asksaveasfilename(
            title=title, defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV files", "*.csv")]
        )
        root.destroy()
        return path if path else default_name
    except Exception as e:
        print(f"(Không mở được hộp thoại lưu file: {e})")
        typed = input(f"{title} - đường dẫn file CSV để lưu (Enter để dùng '{default_name}'): ").strip()
        return typed if typed else default_name


# ----------------------------------------------------------------------------
# Đo lõi: 1 bộ (original, reconstructed, mask) -> list các dict kết quả (1 dict/blob)
# ----------------------------------------------------------------------------
def measure_triplet(original, reconstructed, mask, image_name="", method_name="",
                     pad=6, min_area=3, max_area=400, glcm_size=32):

    if reconstructed.ndim == 3:
        reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY)
    if original.ndim == 3:
        original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Nếu mask không cùng kích thước với ảnh tái tạo (VD: mask tạo từ ảnh gốc độ phân giải khác) -> resize
    if mask.shape != reconstructed.shape:
        mask = cv2.resize(mask, (reconstructed.shape[1], reconstructed.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    if original.shape != reconstructed.shape:
        original = cv2.resize(original, (reconstructed.shape[1], reconstructed.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    H, W = reconstructed.shape
    results = []

    def glcm_feats(patch):
        p = cv2.resize(patch, (glcm_size, glcm_size))
        p = (p.astype(np.float32) / 255 * 15).astype(np.uint8)
        glcm = graycomatrix(p, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                             levels=16, symmetric=True, normed=True)
        return {
            "contrast": float(graycoprops(glcm, "contrast").mean()),
            "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
            "energy": float(graycoprops(glcm, "energy").mean()),
        }

    blob_idx = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        blob_idx += 1
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])

        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, W), min(y + h + pad, H)
        pw, ph = x2 - x1, y2 - y1

        patch_recon = reconstructed[y1:y2, x1:x2]

        # 4 patch tham chiếu cùng kích thước, dịch sang 4 hướng ngay sát patch chính,
        # né vùng mask (không lấy đè lên caliper khác) trong phạm vi có thể
        ref_patches = []
        offsets = [(-pw, 0), (pw, 0), (0, -ph), (0, ph)]
        for dx, dy in offsets:
            rx1, ry1 = x1 + dx, y1 + dy
            rx2, ry2 = rx1 + pw, ry1 + ph
            if 0 <= rx1 and rx2 <= W and 0 <= ry1 and ry2 <= H:
                # Bỏ qua nếu vùng tham chiếu chồng lấn > 10% với mask (tránh lẫn caliper khác)
                overlap = mask_bin[ry1:ry2, rx1:rx2]
                if (overlap > 0).mean() < 0.1:
                    ref_patches.append(reconstructed[ry1:ry2, rx1:rx2])

        if not ref_patches:
            continue  # không đủ vùng tham chiếu hợp lệ, bỏ qua blob này

        ref_means = [float(p.mean()) for p in ref_patches]
        ref_vars = [float(p.var()) for p in ref_patches]
        ref_glcms = [glcm_feats(p) for p in ref_patches]

        recon_var = float(patch_recon.var())
        recon_mean = float(patch_recon.mean())
        recon_glcm = glcm_feats(patch_recon)

        ref_var_avg = float(np.mean(ref_vars))
        ref_contrast_avg = float(np.mean([g["contrast"] for g in ref_glcms]))
        ref_homog_avg = float(np.mean([g["homogeneity"] for g in ref_glcms]))
        ref_energy_avg = float(np.mean([g["energy"] for g in ref_glcms]))

        results.append({
            "image": image_name,
            "method": method_name,
            "blob_id": blob_idx,
            "blob_area_px": int(area),
            "recon_mean": round(recon_mean, 2),
            "recon_var": round(recon_var, 2),
            "ref_mean_avg": round(float(np.mean(ref_means)), 2),
            "ref_var_avg": round(ref_var_avg, 2),
            "var_ratio": round(recon_var / ref_var_avg, 4) if ref_var_avg > 0 else None,
            "recon_glcm_contrast": round(recon_glcm["contrast"], 4),
            "ref_glcm_contrast_avg": round(ref_contrast_avg, 4),
            "contrast_ratio": round(recon_glcm["contrast"] / ref_contrast_avg, 4) if ref_contrast_avg > 0 else None,
            "recon_glcm_homogeneity": round(recon_glcm["homogeneity"], 4),
            "ref_glcm_homogeneity_avg": round(ref_homog_avg, 4),
            "homogeneity_ratio": round(recon_glcm["homogeneity"] / ref_homog_avg, 4) if ref_homog_avg > 0 else None,
            "recon_glcm_energy": round(recon_glcm["energy"], 4),
            "ref_glcm_energy_avg": round(ref_energy_avg, 4),
            "energy_ratio": round(recon_glcm["energy"] / ref_energy_avg, 4) if ref_energy_avg > 0 else None,
        })

    return results


# ----------------------------------------------------------------------------
# Chạy cho 1 bộ ảnh đơn lẻ (CLI)
# ----------------------------------------------------------------------------
def run_single(original_path, reconstructed_path, mask_path, method_name="method", out_csv=None):
    original     = imread_unicode(original_path)
    reconstructed = imread_unicode(reconstructed_path)
    mask         = imread_unicode(mask_path)

    image_name = os.path.basename(reconstructed_path)
    rows = measure_triplet(original, reconstructed, mask, image_name=image_name, method_name=method_name)

    if not rows:
        print("Không phát hiện được blob caliper hợp lệ nào trong mask.")
        return

    for r in rows:
        print(f"[{r['blob_id']}] area={r['blob_area_px']}px | "
              f"var_ratio={r['var_ratio']} | contrast_ratio={r['contrast_ratio']} | "
              f"homogeneity_ratio={r['homogeneity_ratio']} | energy_ratio={r['energy_ratio']}")

    avg_var_ratio = np.mean([r["var_ratio"] for r in rows if r["var_ratio"] is not None])
    avg_contrast_ratio = np.mean([r["contrast_ratio"] for r in rows if r["contrast_ratio"] is not None])
    print(f"\nTrung bình ảnh này -> var_ratio={avg_var_ratio:.4f} | contrast_ratio={avg_contrast_ratio:.4f}")
    print("(Ratio càng gần 1.0 càng tốt — nghĩa là vùng tái tạo giống thống kê với mô lân cận)")

    if out_csv:
        write_csv(rows, out_csv, append=os.path.exists(out_csv))


# ----------------------------------------------------------------------------
# Chạy batch: nhiều ảnh, 1 phương pháp, ghép theo tên file giữa 3 thư mục
# ----------------------------------------------------------------------------
def run_batch(original_dir, reconstructed_dir, mask_dir, method_name, out_csv):
    """
    Yêu cầu: ảnh gốc / ảnh tái tạo / mask phải cùng tên file (đuôi có thể khác nhau,
    sẽ so khớp theo basename không đuôi).
    """
    valid_extensions = ('.png', '.jpg', '.jpeg')

    def _build_index(directory):
        index = {}
        for f in glob.glob(os.path.join(os.path.normpath(directory), "*")):
            if os.path.splitext(f)[1].lower() in valid_extensions:
                stem = os.path.splitext(os.path.basename(f))[0]
                index[stem] = os.path.normpath(f)
        return index

    orig_files  = _build_index(original_dir)
    mask_files  = _build_index(mask_dir)
    recon_files = _build_index(reconstructed_dir)

    common_keys = set(orig_files) & set(mask_files) & set(recon_files)
    print(f"Khớp được {len(common_keys)} ảnh giữa 3 thư mục "
          f"(original={len(orig_files)}, reconstructed={len(recon_files)}, mask={len(mask_files)})")

    missing_in_recon = (set(orig_files) & set(mask_files)) - set(recon_files)
    if missing_in_recon:
        print(f"  Lưu ý: {len(missing_in_recon)} ảnh có original+mask nhưng thiếu file reconstructed, đã bỏ qua.")

    all_rows = []
    for key in sorted(common_keys):
        original      = imread_unicode(orig_files[key])
        reconstructed = imread_unicode(recon_files[key])
        mask          = imread_unicode(mask_files[key])
        if original is None or reconstructed is None or mask is None:
            print(f"  [CẢNH BÁO] Bỏ qua '{key}': không đọc được một hoặc nhiều file ảnh.")
            continue
        rows = measure_triplet(original, reconstructed, mask,
                                image_name=key, method_name=method_name)
        all_rows.extend(rows)

    write_csv(all_rows, out_csv, append=False)
    print(f"\nĐã ghi {len(all_rows)} dòng (mỗi dòng = 1 vùng caliper) vào {out_csv}")

    if all_rows:
        vr = np.mean([r["var_ratio"] for r in all_rows if r["var_ratio"] is not None])
        cr = np.mean([r["contrast_ratio"] for r in all_rows if r["contrast_ratio"] is not None])
        hr = np.mean([r["homogeneity_ratio"] for r in all_rows if r["homogeneity_ratio"] is not None])
        er = np.mean([r["energy_ratio"] for r in all_rows if r["energy_ratio"] is not None])
        print("\n=== TỔNG KẾT TRUNG BÌNH ===")
        print(f"{method_name:10s} | n_blob={len(all_rows):4d} | var_ratio={vr:.3f} | "
              f"contrast_ratio={cr:.3f} | homogeneity_ratio={hr:.3f} | energy_ratio={er:.3f}")
        print("(Ratio càng gần 1.0 càng tốt — nghĩa là vùng tái tạo giống thống kê với mô lân cận)")


def write_csv(rows, path, append=False):
    if not rows:
        return
    mode = "a" if append else "w"
    write_header = not (append and os.path.exists(path))
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # Không truyền tham số nào -> tự mở hộp thoại chọn thư mục, khỏi cần gõ lệnh dài
    if len(sys.argv) == 1:
        print("=== Chế độ chọn thư mục bằng hộp thoại (không cần gõ đường dẫn) ===\n")
        original_dir = pick_folder("Chọn thư mục ẢNH GỐC (còn caliper)")
        reconstructed_dir = pick_folder("Chọn thư mục ẢNH ĐÃ TÁI TẠO (đã xử lý caliper)")
        mask_dir = pick_folder("Chọn thư mục MASK caliper")
        method_name = input("\nTên phương pháp (VD: Gemini, LaMa, Telea) [Enter = 'method']: ").strip() or "method"
        out_csv = pick_save_file("Chọn nơi lưu file kết quả CSV", default_name=f"ket_qua_{method_name}.csv")

        print(f"\nĐang chạy... original={original_dir} | reconstructed={reconstructed_dir} | mask={mask_dir}\n")
        run_batch(original_dir, reconstructed_dir, mask_dir, method_name, out_csv)
        sys.exit(0)

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", action="store_true", help="Chạy chế độ batch nhiều ảnh/nhiều phương pháp")

    # Chế độ đơn lẻ
    parser.add_argument("--original", help="Đường dẫn ảnh gốc (còn caliper)")
    parser.add_argument("--reconstructed", help="Đường dẫn ảnh đã tái tạo")
    parser.add_argument("--mask", help="Đường dẫn binary mask caliper")
    parser.add_argument("--method", default="method", help="Tên phương pháp (Telea/LaMa/Gemini/...)")

    # Chế độ batch (3 thư mục: gốc / tái tạo / mask)
    print("gốc => tái tạo => mask=> output")
    parser.add_argument("--original_dir", help="Thư mục ảnh gốc")
    parser.add_argument("--reconstructed_dir", help="Thư mục ảnh đã tái tạo")
    parser.add_argument("--mask_dir", help="Thư mục mask")

    parser.add_argument("--output", default="inpaint_quality_results.csv", help="File CSV output")
    args = parser.parse_args()

    if args.batch:
        run_batch(args.original_dir, args.reconstructed_dir, args.mask_dir, args.method, args.output)
    else:
        run_single(args.original, args.reconstructed, args.mask, args.method, args.output)
