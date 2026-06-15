"""
merge_labels.py
───────────────
Ghép 2 folder label YOLO:
  - Folder A: label vùng bất thường (class 0)
  - Folder B: label caliper từ makesense.ai
  → Folder C: file đã ghép (class 0 = abnormal, class 1 = caliper)
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox


def pick_folder(title):
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder



def merge(folder_abnormal, folder_caliper, folder_output):
    os.makedirs(folder_output, exist_ok=True)

    all_names = set()
    for f in os.listdir(folder_abnormal):
        if f.endswith(".txt") and f != "classes.txt":
            all_names.add(f)
    for f in os.listdir(folder_caliper):
        if f.endswith(".txt") and f != "classes.txt":
            all_names.add(f)

    count_merged = 0
    count_only_abnormal = 0
    count_only_caliper = 0

    for fname in sorted(all_names):
        path_ab = os.path.join(folder_abnormal, fname)
        path_ca = os.path.join(folder_caliper,  fname)
        path_out = os.path.join(folder_output,  fname)

        lines_ab = []
        lines_ca = []

        # Đọc label abnormal
        if os.path.exists(path_ab):
            with open(path_ab, encoding="utf-8") as f:
                lines_ab = [l.strip() for l in f if l.strip()]

        # Đọc label caliper → đổi class ID thành 1
        if os.path.exists(path_ca):
            with open(path_ca, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    lines_ca.append(line)  # class 1 đã đúng sẵn

        merged = lines_ab + lines_ca

        with open(path_out, "w", encoding="utf-8") as f:
            f.write("\n".join(merged))
            if merged:
                f.write("\n")

        if lines_ab and lines_ca:
            count_merged += 1
        elif lines_ab:
            count_only_abnormal += 1
        elif lines_ca:
            count_only_caliper += 1

    # Ghi classes.txt
    classes_path = os.path.join(folder_output, "classes.txt")
    with open(classes_path, "w") as f:
        f.write("abnormal\ncaliper\n")

    return count_merged, count_only_abnormal, count_only_caliper, len(all_names)


def main():
    print("=" * 55)
    print("  MERGE YOLO LABELS — Abnormal + Caliper")
    print("=" * 55)

    # Chọn folder
    print("\n[1/3] Chọn folder label ABNORMAL (cũ)...")
    folder_ab = pick_folder("Chọn folder label ABNORMAL (class 0)")
    if not folder_ab:
        print("Đã huỷ."); return
    print(f"  → {folder_ab}")

    print("[2/3] Chọn folder label CALIPER (mới từ makesense.ai)...")
    folder_ca = pick_folder("Chọn folder label CALIPER (từ makesense.ai)")
    if not folder_ca:
        print("Đã huỷ."); return
    print(f"  → {folder_ca}")

    print("[3/3] Chọn folder OUTPUT (lưu kết quả ghép)...")
    folder_out = pick_folder("Chọn folder OUTPUT")
    if not folder_out:
        print("Đã huỷ."); return
    print(f"  → {folder_out}")

    # Merge
    print("\nĐang xử lý...")
    m, a, c, total = merge(folder_ab, folder_ca, folder_out)

    print("\n" + "=" * 55)
    print("  KẾT QUẢ")
    print("=" * 55)
    print(f"  Tổng file xử lý    : {total}")
    print(f"  Ghép được cả 2     : {m}  (abnormal + caliper)")
    print(f"  Chỉ có abnormal    : {a}  (chưa label caliper)")
    print(f"  Chỉ có caliper     : {c}  (không có abnormal)")
    print(f"\n  Output lưu tại: {folder_out}")
    print(f"  classes.txt đã tạo: abnormal(0) / caliper(1)")
    print("=" * 55)

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Xong!",
        f"Hoàn thành!\n\n"
        f"Ghép đủ 2 class : {m} file\n"
        f"Chỉ abnormal    : {a} file\n"
        f"Chỉ caliper     : {c} file\n\n"
        f"Output: {folder_out}"
    )
    root.destroy()


if __name__ == "__main__":
    main()
