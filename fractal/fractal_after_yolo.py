"""
Fractal Analysis Module — Phân tích cấu trúc mô siêu âm
=========================================================
Hai thuật toán độc lập, bổ sung cho nhau:

    1. Fractal Dimension (FD) — Differential Box Counting (DBC)
       Đo mức độ PHỨC TẠP của cấu trúc mô trong ROI.
       Tiếp cận intensity-based, không dùng Canny edge,
       để tránh nhiễu speckle đặc trưng của ảnh siêu âm.

    2. Lacunarity — Gliding Box Method (Allain & Cloitre, 1991)
       Đo sự PHÂN BỐ KHÔNG ĐỒNG ĐỀU của texture trong ROI.
       Hai cấu trúc có cùng FD vẫn có thể có Lacunarity khác nhau.

Cả hai trả về float hoặc np.nan (nếu ROI không đủ điều kiện tính toán).
Không bao giờ trả về 0.0 — giá trị đó vô nghĩa trong lý thuyết fractal.

Output không phải là quyết định Kept/Dropped mà là 2 chỉ số độc lập
để phân tích phân phối sau thực nghiệm.
"""

import cv2
import numpy as np


# ─────────────────────────────────────────────
# 1. FRACTAL DIMENSION — Differential Box Counting
# ─────────────────────────────────────────────

def calculate_fd_dbc(roi_image, min_scale_exp=3):
    """
    Tính Fractal Dimension bằng Differential Box Counting (DBC).

    Khác với edge-based (Canny), DBC làm việc trực tiếp trên
    grayscale intensity — coi ROI như một bề mặt 3D (x, y, cường độ).
    Phù hợp hơn với ảnh siêu âm vì không nhạy với speckle noise.

    Tham số min_scale_exp: bỏ qua scale nhỏ hơn 2^min_scale_exp.
        Default = 3 → bỏ scale 2px và 4px (dominated by noise).
        Scale nhỏ nhất được dùng = 8px.

    Parameters
    ----------
    roi_image : np.ndarray
        Ảnh ROI (BGR hoặc grayscale), bất kỳ kích thước nào.
    min_scale_exp : int
        Số mũ tối thiểu của scale box (2^min_scale_exp pixels).

    Returns
    -------
    float : Fractal Dimension (thường 1.0–3.0 cho ảnh 2D)
    np.nan : Nếu ROI quá nhỏ, đồng nhất, hoặc không đủ scale để fit
    """
    # Chuyển sang grayscale
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = roi_image.astype(np.float64)

    h, w = gray.shape
    min_dim = min(h, w)

    # ROI quá nhỏ để có ít nhất 2 scale
    min_size_needed = 2 ** (min_scale_exp + 1)
    if min_dim < min_size_needed:
        return np.nan

    # Kiểm tra ROI đồng nhất (không có texture để đo)
    g_min, g_max = gray.min(), gray.max()
    if g_max == g_min:
        return np.nan

    # Tạo dãy scale: từ 2^min_scale_exp đến 2^p_max
    p_max = int(np.log2(min_dim))
    scales = [2 ** exp for exp in range(min_scale_exp, p_max + 1)]

    if len(scales) < 2:
        return np.nan

    # ── Differential Box Counting ──
    # Với mỗi scale s, chia ảnh thành các ô s×s không chồng lấn.
    # Trong mỗi ô, đếm số "hộp" theo chiều intensity cần để phủ bề mặt:
    #   n(i,j) = floor(max_val / s) - floor(min_val / s) + 1
    # N(s) = tổng n(i,j) trên toàn ảnh
    # FD = slope của log(N) vs log(1/s)
    scale_vals = []
    count_vals = []

    for s in scales:
        total = 0
        for i in range(0, h - s + 1, s):
            for j in range(0, w - s + 1, s):
                patch = gray[i:i + s, j:j + s]
                min_p = patch.min()
                max_p = patch.max()
                # Số hộp cần trong chiều intensity (box height = s)
                n = int(max_p / s) - int(min_p / s) + 1
                total += n

        if total > 0:
            scale_vals.append(s)
            count_vals.append(total)

    if len(scale_vals) < 2:
        return np.nan

    # Linear regression trên log-log space
    x = np.log(1.0 / np.array(scale_vals, dtype=np.float64))
    y = np.log(np.array(count_vals, dtype=np.float64))
    coeffs = np.polyfit(x, y, 1)

    return float(coeffs[0])


# ─────────────────────────────────────────────
# 2. LACUNARITY — Gliding Box Method
# ─────────────────────────────────────────────

def calculate_lacunarity(roi_image, box_sizes=None):
    """
    Tính Lacunarity bằng Gliding Box Method (Allain & Cloitre, 1991).

    Lacunarity đo sự phân bố KHÔNG ĐỒNG ĐỀU của texture trong ROI.
    Hai cấu trúc có thể có FD giống nhau nhưng Lacunarity khác nhau.

        Lacunarity cao  → phân bố cục bộ, hỗn loạn (đặc điểm mô bệnh)
        Lacunarity thấp → phân bố đồng đều (đặc điểm mô bình thường)

    Công thức: Λ(r) = σ²/μ² + 1  (CV bình phương của mass + 1)
    Kết quả cuối = trung bình Λ(r) qua tất cả box_sizes.

    Dùng Integral Image để tính box sum trong O(1) — hiệu quả với dataset lớn.

    Parameters
    ----------
    roi_image : np.ndarray
        Ảnh ROI (BGR hoặc grayscale).
    box_sizes : list[int] | None
        Danh sách kích thước box để glide. Nếu None, tự động chọn
        theo kích thước ROI, bỏ qua box quá nhỏ (<8px).

    Returns
    -------
    float : Lacunarity trung bình (≥ 1.0, không có giới hạn trên)
    np.nan : Nếu ROI không đủ điều kiện hoặc không tính được
    """
    # Chuyển sang grayscale
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray = roi_image.astype(np.float64)

    # Normalize về [0, 1]
    g_min, g_max = gray.min(), gray.max()
    if g_max == g_min:
        return np.nan
    gray_norm = (gray - g_min) / (g_max - g_min)

    h, w = gray_norm.shape

    # Tự động chọn box_sizes phù hợp với kích thước ROI
    if box_sizes is None:
        max_box = min(h, w) // 2
        # Bỏ box < 8px để tránh noise, lấy theo bội số 2
        box_sizes = [s for s in [8, 16, 32, 64, 128] if s <= max_box]

    if not box_sizes:
        return np.nan

    # ── Integral Image (Summed Area Table) ──
    # cv2.integral trả về ảnh (h+1, w+1), cho phép tính
    # tổng bất kỳ vùng chữ nhật trong O(1).
    integral = cv2.integral(gray_norm)  # shape: (h+1, w+1)

    lacunarities = []

    for r in box_sizes:
        n_i = h - r + 1
        n_j = w - r + 1

        if n_i <= 0 or n_j <= 0:
            continue

        # Tính toàn bộ mass matrix cùng lúc (fully vectorized)
        # mass[i, j] = tổng pixel trong box r×r bắt đầu tại (i, j)
        masses = (
            integral[r:h + 1,   r:w + 1]     # góc dưới-phải
          - integral[0:h - r + 1, r:w + 1]   # góc trên-phải
          - integral[r:h + 1,   0:w - r + 1] # góc dưới-trái
          + integral[0:h - r + 1, 0:w - r + 1] # góc trên-trái
        ).flatten()

        mu = masses.mean()
        if mu == 0:
            continue

        # Λ(r) = σ²/μ² + 1
        sigma2 = masses.var()
        lac = (sigma2 / (mu ** 2)) + 1.0
        lacunarities.append(lac)

    if not lacunarities:
        return np.nan

    return float(np.mean(lacunarities))


# ─────────────────────────────────────────────
# 3. CONVENIENCE WRAPPER
# ─────────────────────────────────────────────

def analyze_roi(roi_image, min_scale_exp=3, box_sizes=None):
    """
    Tính cả FD (DBC) và Lacunarity cho một ROI.

    Parameters
    ----------
    roi_image    : np.ndarray — ảnh ROI
    min_scale_exp: int       — scale tối thiểu cho FD (default 3 → 8px)
    box_sizes    : list|None — box sizes cho Lacunarity (None = tự động)

    Returns
    -------
    dict: {
        'fd'        : float | np.nan,
        'lacunarity': float | np.nan
    }
    """
    fd  = calculate_fd_dbc(roi_image, min_scale_exp=min_scale_exp)
    lac = calculate_lacunarity(roi_image, box_sizes=box_sizes)
    return {'fd': fd, 'lacunarity': lac}