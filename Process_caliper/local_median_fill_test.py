"""
inpaint_test.py
===============
So sánh các thuật toán inpainting cho ảnh siêu âm:
  1. Fast Marching Method (FMM)      – baseline OpenCV
  2. Exemplar-based (Criminisi)      – copy patch giống nhất (top_k=1)
  3. Exemplar + Poisson Blending     – như trên + seamlessClone để khớp biên

Dùng: python local_median_fill_test.py
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_file(title: str) -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
    )
    root.destroy()
    if not path:
        raise SystemExit(f"[HUY] Khong chon: {title}")
    return path


def read_img(path: str, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """Đọc ảnh hỗ trợ đường dẫn Unicode."""
    return cv2.imdecode(np.frombuffer(open(path, "rb").read(), np.uint8), flags)


# ---------------------------------------------------------------------------
# Thuật toán 1: Fast Marching (baseline)
# ---------------------------------------------------------------------------

def algo_fmm(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Thuật toán 2: Exemplar-based Inpainting (classic Criminisi, top_k=1)
# ---------------------------------------------------------------------------

def algo_exemplar(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 9,
    search_radius: int = 150,
) -> np.ndarray:
    """
    Exemplar-based Inpainting (Criminisi-style, single best patch).

    Giữ nguyên top_k=1 để BẢO TOÀN TEXTURE speckle của siêu âm.
    Averaging nhiều patch → làm mờ → mất speckle (như median fill).

    Mỗi vòng lặp:
      1. Tìm biên mask (pixel mask tiếp giáp vùng đã biết).
      2. Chọn patch biên có nhiều pixel biết nhất (confidence cao).
      3. Tính SSD map vectorized → tìm patch ứng viên giống nhất.
      4. Copy NGUYÊN 1 patch vào vùng chưa biết → giữ texture.
      5. Lặp đến khi hết mask.
    """
    if patch_size % 2 == 0:
        patch_size += 1
    half = patch_size // 2

    result    = image.copy().astype(np.float32)
    fill_mask = (mask > 127).astype(np.uint8)
    h, w      = result.shape[:2]
    dilate_k  = np.ones((3, 3), np.uint8)
    iteration = 0

    while fill_mask.sum() > 0:
        iteration += 1

        # ── Tìm biên ──────────────────────────────────────────────────────────
        known         = (fill_mask == 0).astype(np.uint8)
        known_dilated = cv2.dilate(known, dilate_k, iterations=1)
        boundary      = (fill_mask == 1) & (known_dilated == 1)
        bpts          = np.argwhere(boundary)

        if len(bpts) == 0:
            print(f"  [WARN] Con {fill_mask.sum()} px nhung het boundary.")
            break

        # ── Chọn patch biên confidence cao nhất ───────────────────────────────
        best_pt   = None
        best_conf = -1

        for pt in bpts:
            py, px = int(pt[0]), int(pt[1])
            if py - half < 0 or py + half >= h or px - half < 0 or px + half >= w:
                continue
            conf = int((fill_mask[py-half : py+half+1, px-half : px+half+1] == 0).sum())
            if conf > best_conf:
                best_conf = conf
                best_pt   = (py, px)

        if best_pt is None:
            fill_mask[bpts[:, 0], bpts[:, 1]] = 0
            continue

        py, px  = best_pt
        t_patch = result   [py-half : py+half+1, px-half : px+half+1].copy()
        t_fill  = fill_mask[py-half : py+half+1, px-half : px+half+1].copy()
        known_px = (t_fill == 0)

        # ── SSD map vectorized ─────────────────────────────────────────────────
        if search_radius:
            sy1 = max(half, py - search_radius)
            sy2 = min(h - half - 1, py + search_radius)
            sx1 = max(half, px - search_radius)
            sx2 = min(w - half - 1, px + search_radius)
        else:
            sy1, sy2 = half, h - half - 1
            sx1, sx2 = half, w - half - 1

        known_flat   = known_px.flatten()
        target_known = t_patch.reshape(-1, 3)[known_flat]

        best_src = None
        best_ssd = np.inf

        if sy2 > sy1 and sx2 > sx1:
            roi      = result   [sy1-half : sy2+half+1, sx1-half : sx2+half+1]
            roi_mask = fill_mask[sy1-half : sy2+half+1, sx1-half : sx2+half+1]

            windows   = sliding_window_view(roi,      (patch_size, patch_size, 3)).squeeze(2)
            mask_wins = sliding_window_view(roi_mask, (patch_size, patch_size))

            valid     = (mask_wins.sum(axis=(-2, -1)) == 0)

            if valid.any():
                cands_flat  = windows.reshape(windows.shape[0], windows.shape[1], -1, 3)
                cands_known = cands_flat[:, :, known_flat, :]
                diff        = cands_known - target_known[np.newaxis, np.newaxis]
                ssd_map     = (diff ** 2).sum(axis=(-2, -1)).astype(np.float64)
                ssd_map[~valid] = np.inf

                min_idx  = np.unravel_index(np.argmin(ssd_map), ssd_map.shape)
                best_ssd = float(ssd_map[min_idx])
                best_src = (sy1 + int(min_idx[0]), sx1 + int(min_idx[1]))

        # Fallback brute-force
        if best_src is None:
            for r in range(half, h - half):
                for c in range(half, w - half):
                    if fill_mask[r-half:r+half+1, c-half:c+half+1].sum() > 0:
                        continue
                    cand = result[r-half:r+half+1, c-half:c+half+1].reshape(-1, 3)[known_flat]
                    ssd  = float(((target_known - cand) ** 2).sum())
                    if ssd < best_ssd:
                        best_ssd = ssd
                        best_src = (r, c)

        if best_src is None:
            fill_mask[py, px] = 0
            continue

        # ── Copy 1 patch tốt nhất (top_k=1 → giữ nguyên texture) ─────────────
        sr, sc    = best_src
        src_patch = result[sr-half : sr+half+1, sc-half : sc+half+1]
        unknown   = (t_fill == 1)
        result   [py-half : py+half+1, px-half : px+half+1][unknown] = src_patch[unknown]
        fill_mask[py-half : py+half+1, px-half : px+half+1]          = 0

        remaining = int(fill_mask.sum())
        if iteration % 5 == 0 or remaining == 0:
            print(f"  EXP iter {iteration:3d}: remaining={remaining:5d} px  ssd={best_ssd:.1f}")

    print(f"  Exemplar xong sau {iteration} iterations.")
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Post-process: Feathered Boundary (blur-based, không dùng FMM)
# ---------------------------------------------------------------------------

def feather_blend(filled: np.ndarray, mask: np.ndarray, feather_px: int = 8) -> np.ndarray:
    """
    Làm mịn seam ở biên mask bằng cách blend exemplar với phiên bản
    Gaussian blur của CHÍNH NÓ — không dùng FMM, không dùng ảnh gốc.

    Nguyên lý:
      dist = khoảng cách từng pixel trong mask đến biên mask
      alpha = clip(dist / feather_px, 0, 1)
        * alpha=0 ở biên      → dùng blurred_exemplar (mịn)
        * alpha=1 ở trung tâm → dùng exemplar gốc (texture đầy đủ)

    Ưu điểm so với dùng FMM làm reference:
      - Không mang "rỗng/mịn" của FMM vào
      - Vẫn là texture siêu âm (chỉ blur nhẹ ở biên)
      - Seam giảm mà không mất speckle bên trong
    """
    binary = (mask > 127).astype(np.uint8)
    dist   = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    alpha  = np.clip(dist / max(feather_px, 1), 0.0, 1.0)[:, :, np.newaxis]

    # Reference = blur nhẹ của chính exemplar (sigma = feather_px/2)
    sigma   = max(feather_px / 2.0, 1.0)
    blurred = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma, sigmaY=sigma)

    f   = filled.astype(np.float32)
    ref = blurred.astype(np.float32)

    blend = f * alpha + ref * (1.0 - alpha)
    out   = f.copy()
    out[binary == 1] = blend[binary == 1]   # chỉ áp dụng trong vùng mask

    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Visualize
# ---------------------------------------------------------------------------

def make_comparison(original, fmm, exp, mask) -> np.ndarray:
    """4 panel: Original | Mask | FMM | Exemplar."""
    overlay  = original.copy()
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    red_tint = np.zeros_like(overlay)
    red_tint[:, :, 2] = 180
    overlay  = np.where(mask_rgb > 0,
                        cv2.addWeighted(overlay, 0.5, red_tint, 0.5, 0),
                        overlay).astype(np.uint8)

    cfg = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
               color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    def lb(img, txt):
        out = img.copy()
        cv2.putText(out, txt, (6, 20), **cfg)
        return out

    return np.hstack([
        lb(original, "Original"),
        lb(overlay,  "Mask"),
        lb(fmm,      "Fast Marching"),
        lb(exp,      "Exemplar (Criminisi)"),
    ])


def show_and_save(img: np.ndarray, save_path: str = None):
    win = "Comparison  (press any key to close)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w  = img.shape[:2]
    scale = min(1920 / w, 1000 / h, 1.0)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))
    cv2.imshow(win, img)
    if save_path:
        ext = "." + save_path.rsplit(".", 1)[-1]
        _, buf = cv2.imencode(ext, img)
        open(save_path, "wb").write(buf.tobytes())
        print(f"  Luu: {save_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Inpainting Comparison Test ===\n")

    img_path  = pick_file("Chon ANH GOC (sieu am)")
    mask_path = pick_file("Chon ANH MASK (trang/den)")

    original = read_img(img_path,  cv2.IMREAD_COLOR)
    mask     = read_img(mask_path, cv2.IMREAD_GRAYSCALE)

    if original is None or mask is None:
        raise SystemExit("Loi: Khong doc duoc anh.")
    if original.shape[:2] != mask.shape[:2]:
        raise SystemExit(f"Loi: Kich thuoc khong khop {original.shape[:2]} vs {mask.shape[:2]}")

    print(f"  Anh : {img_path.split('/')[-1]}  ({original.shape[1]}x{original.shape[0]})")
    print(f"  Mask: {mask_path.split('/')[-1]}  — {int((mask>127).sum())} px can lap\n")

    # ── Tham số ──────────────────────────────────────────────────────────────
    FMM_RADIUS  = 3    # bán kính Fast Marching
    EXP_PATCH   = 9    # kích thước patch Exemplar (px, nên lẻ: 7/9/11)
    EXP_RADIUS  = 150  # bán kính tìm kiếm (px); None = toàn ảnh
    # ─────────────────────────────────────────────────────────────────────────

    print("[1/2] Fast Marching Method (FMM)...")
    fmm = algo_fmm(original, mask, radius=FMM_RADIUS)
    print("  Xong.\n")

    print(f"[2/2] Exemplar-based Inpainting (patch={EXP_PATCH}, radius={EXP_RADIUS})...")
    exp = algo_exemplar(original, mask, patch_size=EXP_PATCH, search_radius=EXP_RADIUS)
    print("  Xong.\n")

    comparison = make_comparison(original, fmm, exp, mask)

    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    save_path = filedialog.asksaveasfilename(
        title="Luu anh so sanh (bo qua de chi xem)",
        defaultextension=".png",
        filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
    )
    root.destroy()

    show_and_save(comparison, save_path if save_path else None)
