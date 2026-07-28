import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

# ──────────────────────────────────────────────
# Wrapper đọc/ghi ảnh hỗ trợ đường dẫn Unicode
# (OpenCV trên Windows không hỗ trợ ký tự đặc biệt)
# ──────────────────────────────────────────────
def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Đọc ảnh từ đường dẫn có chứa ký tự Unicode (tiếng Việt, v.v.)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, flags)
        return img
    except Exception as e:
        print(f"[LỖI] imread_unicode thất bại cho '{path}': {e}")
        return None

def imwrite_unicode(path, img):
    """Ghi ảnh ra đường dẫn có chứa ký tự Unicode."""
    try:
        ext = os.path.splitext(path)[1].lower()
        success, buf = cv2.imencode(ext, img)
        if not success:
            print(f"[LỖI] imwrite_unicode: không thể encode ảnh sang '{ext}'")
            return False
        buf.tofile(path)
        return True
    except Exception as e:
        print(f"[LỖI] imwrite_unicode thất bại cho '{path}': {e}")
        return False

# ──────────────────────────────────────────────

def find_file_any_ext(directory, stem):
    """Tìm file theo tên (không phân biệt phần mở rộng) trong thư mục."""
    valid_extensions = ('.png', '.jpg', '.jpeg')
    for ext in valid_extensions:
        candidate = os.path.normpath(os.path.join(directory, stem + ext))
        if os.path.exists(candidate):
            return candidate
    return None


def build_feather_mask(mask_binary, dilate_px=15, blur_px=31):
    """
    Tạo alpha mask mờ dần (feathered) từ mask nhị phân gốc.

    Quy trình:
      1. Dilate mask gốc ra ngoài `dilate_px` pixel → vùng lấy ảnh tái tạo rộng hơn.
      2. GaussianBlur mask đã dilate → chuyển biên cứng thành gradient mờ dần.
      3. Chuẩn hóa về [0.0, 1.0] để dùng làm hệ số blend.

    Tham số:
        mask_binary : np.ndarray, uint8, single-channel, giá trị 0 hoặc 255.
        dilate_px   : Số pixel mở rộng ra ngoài biên mask gốc.
        blur_px     : Kích thước kernel Gaussian (phải là số lẻ).
                      Giá trị lớn hơn → vùng chuyển tiếp rộng hơn và mượt hơn.

    Trả về:
        alpha : np.ndarray float32, shape (H, W), giá trị trong [0, 1].
                1.0 = hoàn toàn lấy ảnh tái tạo, 0.0 = hoàn toàn lấy ảnh gốc.
    """
    # Đảm bảo blur_px là số lẻ ≥ 1
    blur_px = max(1, blur_px | 1)

    # Bước 1: Dilate để mở rộng vùng mask
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1)
    )
    mask_dilated = cv2.dilate(mask_binary, kernel, iterations=1)

    # Bước 2: Gaussian blur tạo gradient mờ dần ở biên
    mask_blurred = cv2.GaussianBlur(
        mask_dilated.astype(np.float32),
        (blur_px, blur_px),
        sigmaX=0  # OpenCV tự tính sigma từ kích thước kernel
    )

    # Bước 3: Chuẩn hóa về [0, 1]
    alpha = mask_blurred / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def blend_with_feather(orig_img, recon_img, alpha):
    """
    Blend ảnh gốc và ảnh tái tạo theo alpha mask mờ dần.

    result = alpha * recon + (1 - alpha) * orig

    Tham số:
        orig_img  : np.ndarray uint8 BGR
        recon_img : np.ndarray uint8 BGR
        alpha     : np.ndarray float32 shape (H, W), giá trị [0, 1]

    Trả về:
        result : np.ndarray uint8 BGR
    """
    alpha_3c = alpha[:, :, np.newaxis]  # broadcast sang 3 kênh màu
    orig_f   = orig_img.astype(np.float32)
    recon_f  = recon_img.astype(np.float32)

    blended = alpha_3c * recon_f + (1.0 - alpha_3c) * orig_f
    return np.clip(blended, 0, 255).astype(np.uint8)


def merge_reconstructed_images(
    orig_dir, recon_dir, mask_dir, output_dir,
    dilate_px=15, blur_px=31
):
    """
    Merge ảnh gốc với ảnh tái tạo sử dụng kỹ thuật feathered blending.

    Tham số:
        orig_dir   : Thư mục ảnh gốc.
        recon_dir  : Thư mục ảnh sau tái tạo (AI inpaint).
        mask_dir   : Thư mục mask nhị phân (vùng cần thay thế = trắng).
        output_dir : Thư mục lưu ảnh kết quả.
        dilate_px  : Số pixel mở rộng mask trước khi blur (mặc định 15).
        blur_px    : Kích thước kernel Gaussian blur (mặc định 31, số lẻ).
    """
    # Chuẩn hóa tất cả đường dẫn thư mục
    orig_dir   = os.path.normpath(orig_dir)
    recon_dir  = os.path.normpath(recon_dir)
    mask_dir   = os.path.normpath(mask_dir)
    output_dir = os.path.normpath(output_dir)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"[LỖI] Không thể tạo thư mục output '{output_dir}': {e}")
        return

    try:
        valid_extensions = ('.png', '.jpg', '.jpeg')
        filenames = [f for f in os.listdir(orig_dir) if f.lower().endswith(valid_extensions)]
    except Exception as e:
        print(f"[LỖI] Không thể đọc danh sách file từ thư mục gốc '{orig_dir}': {e}")
        return

    if not filenames:
        print(f"[CẢNH BÁO] Không tìm thấy ảnh nào trong thư mục gốc: '{orig_dir}'")
        return

    for filename in filenames:
        stem     = os.path.splitext(filename)[0]
        orig_ext = os.path.splitext(filename)[1].lower()

        orig_path   = os.path.normpath(os.path.join(orig_dir, filename))
        output_path = os.path.normpath(os.path.join(output_dir, stem + orig_ext))

        recon_path = find_file_any_ext(recon_dir, stem)
        if recon_path is None:
            print(f"[CẢNH BÁO] Không tìm thấy ảnh tái tạo cho '{stem}' (thử .png/.jpg/.jpeg), bỏ qua.")
            continue

        mask_path = find_file_any_ext(mask_dir, stem)
        if mask_path is None:
            print(f"[CẢNH BÁO] Không tìm thấy mask cho '{stem}' (thử .png/.jpg/.jpeg), bỏ qua.")
            continue

        print(f"[INFO] Xử lý: orig='{filename}' | recon='{os.path.basename(recon_path)}' | mask='{os.path.basename(mask_path)}'")

        try:
            orig_img = imread_unicode(orig_path)
            if orig_img is None:
                print(f"[LỖI] Không thể đọc ảnh gốc: '{orig_path}'")
                continue

            recon_img = imread_unicode(recon_path)
            if recon_img is None:
                print(f"[LỖI] Không thể đọc ảnh tái tạo: '{recon_path}'")
                continue

            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[LỖI] Không thể đọc mask: '{mask_path}'")
                continue
        except Exception as e:
            print(f"[LỖI] Lỗi khi đọc ảnh cho '{filename}': {e}")
            continue

        try:
            # ── Resize về cùng kích thước ảnh gốc nếu cần ──
            if orig_img.shape[:2] != mask.shape[:2]:
                print(f"[THÔNG BÁO] Resize mask để khớp kích thước ảnh gốc cho '{filename}'.")
                mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            if orig_img.shape[:2] != recon_img.shape[:2]:
                print(f"[THÔNG BÁO] Resize ảnh tái tạo để khớp kích thước ảnh gốc cho '{filename}'.")
                recon_img = cv2.resize(recon_img, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_CUBIC)

            # ── Nhị phân hóa mask ──
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # ── Tạo feathered alpha mask (mở rộng + blur biên) ──
            alpha = build_feather_mask(mask_binary, dilate_px=dilate_px, blur_px=blur_px)

            # ── Blend mờ dần thay vì ghi đè cứng ──
            result_img = blend_with_feather(orig_img, recon_img, alpha)
        except Exception as e:
            print(f"[LỖI] Lỗi khi xử lý ảnh '{filename}': {e}")
            continue

        try:
            ok = imwrite_unicode(output_path, result_img)
            if ok:
                print(f"[OK] Đã lưu kết quả: '{output_path}'")
            else:
                print(f"[LỖI] Ghi file thất bại: '{output_path}'")
        except Exception as e:
            print(f"[LỖI] Không thể lưu ảnh output '{output_path}': {e}")

def select_folder(title):
    try:
        root = tk.Tk()
        root.withdraw()
        folder = filedialog.askdirectory(title=title)
        root.destroy()
        return folder
    except Exception as e:
        print(f"[LỖI] Lỗi khi mở hộp thoại chọn thư mục ('{title}'): {e}")
        return None

if __name__ == '__main__':
    try:
        print("gốc => tái tạo => mask => output")

        folder_anh_goc = select_folder("Chọn folder ảnh gốc")
        if not folder_anh_goc:
            print("[LỖI] Chưa chọn thư mục ảnh gốc. Thoát chương trình.")
            exit(1)
        print("goc done")

        folder_anh_sau_tai_tao = select_folder("Chọn folder ảnh sau tái tạo")
        if not folder_anh_sau_tai_tao:
            print("[LỖI] Chưa chọn thư mục ảnh tái tạo. Thoát chương trình.")
            exit(1)
        print("tai tao done")

        folder_mask_nhi_phan = select_folder("Chọn folder mask nhị phân")
        if not folder_mask_nhi_phan:
            print("[LỖI] Chưa chọn thư mục mask nhị phân. Thoát chương trình.")
            exit(1)
        print("nhi phan done")

        folder_output = select_folder("Chọn folder lưu output")
        if not folder_output:
            print("[LỖI] Chưa chọn thư mục output. Thoát chương trình.")
            exit(1)

        # ── Tham số blend — điều chỉnh tại đây nếu cần ──
        # dilate_px : mở rộng mask bao nhiêu pixel ra ngoài biên gốc
        # blur_px   : kernel Gaussian blur (phải là số lẻ); lớn hơn = mờ hơn
        merge_reconstructed_images(
            folder_anh_goc,
            folder_anh_sau_tai_tao,
            folder_mask_nhi_phan,
            folder_output,
            dilate_px=15,
            blur_px=31,
        )
        print("[HOÀN THÀNH] Đã xử lý xong tất cả ảnh.")

    except Exception as e:
        print(f"[LỖI NGHIÊM TRỌNG] Đã xảy ra lỗi không mong muốn: {e}")