"""
So sánh hiệu quả model giữa lúc TRAIN (val set) và lúc TEST (test set riêng).

Cách dùng:
1. Chạy: python compare_train_test.py
2. Lần lượt chọn file/thư mục qua cửa sổ hộp thoại:
      • best.pt        — file trọng số tốt nhất
      • results.csv    — file kết quả train của Ultralytics
      • Thư mục output — nơi lưu kết quả
3. Kết quả:
      - test_metrics.json       → chỉ số đầy đủ trên test set
      - comparison_summary.csv  → bảng so sánh val/test
      - comparison_charts.png   → biểu đồ so sánh trực quan

Lưu ý: DATA_YAML được cố định bên dưới vì không thay đổi.
"""

import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# Fix encoding cho terminal Windows (CP1252 không hỗ trợ tiếng Việt)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIG CỐ ĐỊNH — chỉ sửa khi chuyển máy
# ─────────────────────────────────────────────
DATA_YAML = Path(r"D:\Code_for_set\ADSUS\data.yaml")

# ─────────────────────────────────────────────
# HELPERS: HỘP THOẠI CHỌN FILE / THƯ MỤC
# ─────────────────────────────────────────────

def _make_root():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    return root


def pick_file(title, filetypes=(("All files", "*.*"),)):
    """Mở hộp thoại chọn file. Trả về Path hoặc None nếu huỷ."""
    root = _make_root()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return Path(path) if path else None


def pick_folder(title):
    """Mở hộp thoại chọn thư mục. Trả về Path hoặc None nếu huỷ."""
    root = _make_root()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return Path(folder) if folder else None


def pick_all_paths():
    """
    Hỏi người dùng lần lượt 3 mục cần thiết.
    data.yaml được cố định — không cần chọn.
    Trả về (best_pt, results_csv, output_dir) hoặc None nếu huỷ.
    """
    print("=" * 60)
    print("  COMPARE TRAIN vs TEST — Chọn file/thư mục")
    print("=" * 60)
    print(f"  data.yaml : {DATA_YAML}  (cố định)")

    # 1. best.pt
    print("\n[1/3] Chọn file best.pt ...")
    best_pt = pick_file(
        "1/3 — Chọn file best.pt (weights)",
        filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")],
    )
    if not best_pt:
        print("  → Đã huỷ."); return None
    print(f"  best.pt   : {best_pt}")

    # 2. results.csv
    print("\n[2/3] Chọn file results.csv từ lần train ...")
    results_csv = pick_file(
        "2/3 — Chọn file results.csv (training log)",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    if not results_csv:
        print("  → Đã huỷ."); return None
    print(f"  results   : {results_csv}")

    # 3. output dir
    print("\n[3/3] Chọn thư mục lưu kết quả ...")
    output_dir = pick_folder("3/3 — Chọn thư mục lưu kết quả")
    if not output_dir:
        print("  → Đã huỷ."); return None
    print(f"  output    : {output_dir}")
    print()

    return best_pt, results_csv, output_dir


# ─────────────────────────────────────────────
# BƯỚC 1: EVAL TRÊN VAL SET VÀ TEST SET (chỉ class 0)
# ─────────────────────────────────────────────

def _run_eval(best_pt, data_yaml, split, output_dir):
    """
    Chạy model.val() trên một split cụ thể, CHỈ class 0.
    Trả về dict các chỉ số và lưu JSON.
    """
    from ultralytics import YOLO

    label = split.upper()
    print(f"\n[►] Đang chạy evaluation trên {label} set (class 0 only) ...")
    model = YOLO(str(best_pt))

    # classes=[0] → chỉ tính metrics cho class 0
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        classes=[0],
        save_json=(split == "test"),
    )

    result = {
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "fitness":   float(metrics.fitness) if hasattr(metrics, "fitness") else None,
    }

    json_path = output_dir / f"{split}_metrics_class0.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== KẾT QUẢ TRÊN {label} SET (class 0) ===")
    for k, v in result.items():
        print(f"  {k:<12}: {v}")

    return result


def run_val_eval(best_pt, data_yaml, output_dir):
    """Đánh giá best.pt trên VAL split, chỉ class 0."""
    return _run_eval(best_pt, data_yaml, "val", output_dir)


def run_test_eval(best_pt, data_yaml, output_dir):
    """Đánh giá best.pt trên TEST split, chỉ class 0."""
    return _run_eval(best_pt, data_yaml, "test", output_dir)


# ─────────────────────────────────────────────
# BƯỚC 2: ĐỌC KẾT QUẢ TRAIN TỪ results.csv
# ─────────────────────────────────────────────

# Tên cột metric cần dùng — thứ tự ưu tiên (Ultralytics thay đổi qua các version)
_COL_CANDIDATES = {
    "precision":  ["metrics/precision(B)", "metrics/precision"],
    "recall":     ["metrics/recall(B)",    "metrics/recall"],
    "mAP50":      ["metrics/mAP50(B)",     "metrics/mAP50"],
    "mAP50-95":   ["metrics/mAP50-95(B)",  "metrics/mAP50-95"],
    "train/cls":  ["train/cls_loss",       "train/cls"],
    "train/box":  ["train/box_loss",       "train/box"],
    "train/dfl":  ["train/dfl_loss",       "train/dfl"],
    "val/cls":    ["val/cls_loss",         "val/cls"],
    "val/box":    ["val/box_loss",         "val/box"],
    "val/dfl":    ["val/dfl_loss",         "val/dfl"],
}


def _find_col(df_cols, candidates):
    """Tìm tên cột đầu tiên khớp trong danh sách ứng viên."""
    for c in candidates:
        if c in df_cols:
            return c
    return None


def load_train_val_summary(results_csv):
    """Đọc results.csv, trả về summary dict và DataFrame gốc."""
    # Thử các encoding phổ biến: utf-8 → cp1252 (Windows) → latin-1
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(results_csv, encoding=enc)
            break
        except (UnicodeDecodeError, Exception):
            continue
    else:
        raise ValueError(f"Không đọc được results.csv: {results_csv}")
    df.columns = [c.strip() for c in df.columns]

    print(f"\n[i] Cột trong results.csv: {list(df.columns)}")

    # Tìm cột mAP50-95 để xác định best epoch
    map_col = _find_col(df.columns, _COL_CANDIDATES["mAP50-95"])
    if map_col is None:
        raise KeyError(
            f"Không tìm thấy cột mAP50-95 trong results.csv.\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    last_row = df.iloc[-1]
    best_idx = df[map_col].idxmax()
    best_row = df.loc[best_idx]

    def _get(row, key):
        col = _find_col(df.columns, _COL_CANDIDATES[key])
        return float(row[col]) if col else None

    summary = {
        "last_epoch": {
            "epoch":     int(last_row.get("epoch", len(df) - 1)),
            "precision": _get(last_row, "precision"),
            "recall":    _get(last_row, "recall"),
            "mAP50":     _get(last_row, "mAP50"),
            "mAP50-95":  _get(last_row, "mAP50-95"),
        },
        "best_epoch": {
            "epoch":     int(best_row.get("epoch", best_idx)),
            "precision": _get(best_row, "precision"),
            "recall":    _get(best_row, "recall"),
            "mAP50":     _get(best_row, "mAP50"),
            "mAP50-95":  _get(best_row, "mAP50-95"),
        },
    }
    return summary, df


# ─────────────────────────────────────────────
# BƯỚC 3: BẢNG SO SÁNH
# ─────────────────────────────────────────────

def build_comparison_table(val_metrics, test_metrics, output_dir):
    """
    Tạo bảng so sánh Val vs Test — cả hai đều dùng best.pt, chỉ class 0.
    Lưu CSV.
    """
    keys = ("precision", "recall", "mAP50", "mAP50-95")
    rows = [
        {"split": "Val  (best.pt | class 0)", **{k: val_metrics[k]  for k in keys}},
        {"split": "Test (best.pt | class 0)", **{k: test_metrics[k] for k in keys}},
    ]

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(output_dir / "comparison_summary.csv", index=False)

    print("\n=== BẢNG SO SÁNH (class 0 only) ===")
    print(comp_df.to_string(index=False))
    return comp_df


# ─────────────────────────────────────────────
# BƯỚC 4: VẼ BIỂU ĐỒ
# ─────────────────────────────────────────────

def _fmt(v):
    return f"{v:.3f}" if v is not None else "N/A"


def plot_comparison(df_curve, val_metrics, test_metrics, output_dir):
    """Vẽ 5 biểu đồ: mAP50, mAP50-95, cls/box/dfl loss, bar chart tổng quan."""
    cols = df_curve.columns
    epochs = df_curve.get("epoch", pd.RangeIndex(len(df_curve)))

    # Màu sắc nhất quán
    C_VAL   = "#1D9E75"
    C_TEST  = "#D85A30"
    C_TRAIN = "#378ADD"
    C_GRID  = "#888888"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("So sánh Train / Val / Test — YOLO  [Báo cáo: class 0]", fontsize=15, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # ── 1. mAP50 ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    map50_col = _find_col(cols, _COL_CANDIDATES["mAP50"])
    if map50_col:
        ax1.plot(epochs, df_curve[map50_col], color=C_VAL, label="Val mAP50 (epoch)")
    ax1.axhline(test_metrics["mAP50"], color=C_TEST, linestyle="--",
                label=f"Test mAP50 = {_fmt(test_metrics['mAP50'])} (class 0)")
    best_map50 = val_metrics["mAP50"]
    if best_map50:
        ax1.axhline(best_map50, color=C_GRID, linestyle=":", linewidth=1,
                    label=f"Val best.pt = {_fmt(best_map50)} (class 0)")
    ax1.set_title("mAP50: Val vs Test"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("mAP50")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # ── 2. mAP50-95 ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    map95_col = _find_col(cols, _COL_CANDIDATES["mAP50-95"])
    if map95_col:
        ax2.plot(epochs, df_curve[map95_col], color="#7F77DD", label="Val mAP50-95 (epoch)")
    ax2.axhline(test_metrics["mAP50-95"], color=C_TEST, linestyle="--",
                label=f"Test mAP50-95 = {_fmt(test_metrics['mAP50-95'])} (class 0)")
    best_map95 = val_metrics["mAP50-95"]
    if best_map95:
        ax2.axhline(best_map95, color=C_GRID, linestyle=":", linewidth=1,
                    label=f"Val best.pt = {_fmt(best_map95)} (class 0)")
    ax2.set_title("mAP50-95: Val vs Test"); ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP50-95")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # ── 3. cls_loss ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    tcls = _find_col(cols, _COL_CANDIDATES["train/cls"])
    vcls = _find_col(cols, _COL_CANDIDATES["val/cls"])
    if tcls: ax3.plot(epochs, df_curve[tcls], color=C_TRAIN, label="train cls_loss")
    if vcls: ax3.plot(epochs, df_curve[vcls], color=C_TEST,  label="val cls_loss", linestyle="--")
    ax3.set_title("Cls Loss: Train vs Val"); ax3.set_xlabel("Epoch"); ax3.set_ylabel("Loss")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

    # ── 4. box_loss ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    tbox = _find_col(cols, _COL_CANDIDATES["train/box"])
    vbox = _find_col(cols, _COL_CANDIDATES["val/box"])
    if tbox: ax4.plot(epochs, df_curve[tbox], color=C_TRAIN, label="train box_loss")
    if vbox: ax4.plot(epochs, df_curve[vbox], color=C_TEST,  label="val box_loss", linestyle="--")
    ax4.set_title("Box Loss: Train vs Val"); ax4.set_xlabel("Epoch"); ax4.set_ylabel("Loss")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    # ── 5. dfl_loss ───────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    tdfl = _find_col(cols, _COL_CANDIDATES["train/dfl"])
    vdfl = _find_col(cols, _COL_CANDIDATES["val/dfl"])
    if tdfl: ax5.plot(epochs, df_curve[tdfl], color=C_TRAIN, label="train dfl_loss")
    if vdfl: ax5.plot(epochs, df_curve[vdfl], color=C_TEST,  label="val dfl_loss", linestyle="--")
    ax5.set_title("DFL Loss: Train vs Val"); ax5.set_xlabel("Epoch"); ax5.set_ylabel("Loss")
    ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    # ── 6. Bar chart: Val best vs Test ────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    metric_names = ["precision", "recall", "mAP50", "mAP50-95"]
    val_vals  = [val_metrics[m]  or 0 for m in metric_names]
    test_vals = [test_metrics[m] or 0 for m in metric_names]

    x     = range(len(metric_names))
    width = 0.35
    bars1 = ax6.bar([i - width/2 for i in x], val_vals,  width,
                    label="Val  (best.pt | class 0)", color=C_VAL)
    bars2 = ax6.bar([i + width/2 for i in x], test_vals, width,
                    label="Test (best.pt | class 0)", color=C_TEST)

    # Ghi số lên đầu cột
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    ax6.set_xticks(list(x))
    ax6.set_xticklabels(metric_names, fontsize=9)
    ax6.set_title("Nghiệm thu: Val vs Test (class 0 only)")
    ax6.set_ylim(0, 1.12)
    ax6.legend(fontsize=8)
    ax6.grid(axis="y", alpha=0.3)

    # Lưu file
    out_path = output_dir / "comparison_charts.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[✓] Đã lưu biểu đồ: {out_path}")

    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # Bước 0: Chọn đường dẫn qua hộp thoại
    paths = pick_all_paths()
    if paths is None:
        print("Đã huỷ — không có file nào được chọn.")
        sys.exit(0)

    best_pt, results_csv, output_dir = paths
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bước 1a: Eval VAL set — best.pt, chỉ class 0 (số liệu nghiệm thu)
    val_metrics = run_val_eval(best_pt, DATA_YAML, output_dir)

    # Bước 1b: Eval TEST set — best.pt, chỉ class 0 (số liệu nghiệm thu)
    test_metrics = run_test_eval(best_pt, DATA_YAML, output_dir)

    # Bước 2: Đọc training log (dùng để vẽ đường cong overfitting)
    train_val_summary, df_curve = load_train_val_summary(results_csv)

    # Bước 3: Bảng so sánh Val vs Test (class 0 only)
    build_comparison_table(val_metrics, test_metrics, output_dir)

    # Bước 4: Biểu đồ
    plot_comparison(df_curve, val_metrics, test_metrics, output_dir)

    # Thông báo hoàn tất
    root = _make_root()
    messagebox.showinfo(
        "Hoàn thành!",
        f"So sánh hoàn tất!\n\n"
        f"Kết quả lưu tại:\n{output_dir}\n\n"
        f"  • val_metrics_class0.json\n"
        f"  • test_metrics_class0.json\n"
        f"  • comparison_summary.csv\n"
        f"  • comparison_charts.png",
    )
    root.destroy()

    print(f"\n[✓] Hoàn tất! Xem kết quả tại: {output_dir}")


if __name__ == "__main__":
    main()