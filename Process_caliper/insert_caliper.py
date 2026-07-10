import os
import cv2
import numpy as np
import random

def paste_calipers_on_normal(bg_path, marker_x_path, marker_plus_path, output_path):
    bg_img = cv2.imread(bg_path)
    m_x = cv2.imread(marker_x_path, cv2.IMREAD_UNCHANGED)
    m_p = cv2.imread(marker_plus_path, cv2.IMREAD_UNCHANGED)

    if bg_img is None or m_x is None or m_p is None:
        return False

    bg_h, bg_w = bg_img.shape[:2]
    mx_h, mx_w = m_x.shape[:2]
    mp_h, mp_w = m_p.shape[:2]

    tumor_w = random.randint(40, 120)
    tumor_h = random.randint(30, 90)

    margin_x = 100
    margin_y = 150
    
    if bg_w - 2 * margin_x <= tumor_w or bg_h - 2 * margin_y <= tumor_h:
        return False
        
    center_x = random.randint(margin_x + tumor_w // 2, bg_w - margin_x - tumor_w // 2)
    center_y = random.randint(margin_y + tumor_h // 2, bg_h - margin_y - tumor_h // 2)

    plus_left_center  = (center_x - tumor_w // 2, center_y)
    plus_right_center = (center_x + tumor_w // 2, center_y)
    x_top_center    = (center_x, center_y - tumor_h // 2)
    x_bottom_center = (center_x, center_y + tumor_h // 2)

    caliper_configs = [
        (plus_left_center[0], plus_left_center[1], m_p, mp_h, mp_w),
        (plus_right_center[0], plus_right_center[1], m_p, mp_h, mp_w),
        (x_top_center[0], x_top_center[1], m_x, mx_h, mx_w),
        (x_bottom_center[0], x_bottom_center[1], m_x, mx_h, mx_w)
    ]

    for cx, cy, marker, mh, mw in caliper_configs:
        x_left = cx - mw // 2
        y_top = cy - mh // 2

        if x_left < 0 or y_top < 0 or (x_left + mw) > bg_w or (y_top + mh) > bg_h:
            continue

        alpha = marker[:, :, 3] / 255.0
        marker_rgb = marker[:, :, :3]

        roi = bg_img[y_top:y_top+mh, x_left:x_left+mw]

        for c in range(3):
            roi[:, :, c] = (alpha * marker_rgb[:, :, c] + (1.0 - alpha) * roi[:, :, c])

        bg_img[y_top:y_top+mh, x_left:x_left+mw] = roi

    cv2.imwrite(output_path, bg_img)
    return True

def process_batch_dir(input_dir, marker_x_path, marker_plus_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ['.jpg', '.jpeg', '.png']
    all_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    count = 0
    for img_name in all_files:
        bg_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, f"{img_name}")
        
        success = paste_calipers_on_normal(bg_path, marker_x_path, marker_plus_path, output_path)
        if success:
            count += 1

    print(f"[+] Hoan tat! Da them caliper cho {count}/{len(all_files)} anh.")
    print(f"[+] Thu muc ket qua: {output_dir}")

if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog, messagebox

    MARKER_X    = r'D:\AI_Data\Process_caliper\cut_caliper\caliperx.png'
    MARKER_PLUS = r'D:\AI_Data\Process_caliper\cut_caliper\caliper+.png'

    root = tk.Tk()
    root.withdraw()

    print(">>> Hay chon folder INPUT (thu muc anh goc)...")
    INPUT_NORMAL_DIR = filedialog.askdirectory(title="Chon thu muc anh goc (input)")
    if not INPUT_NORMAL_DIR:
        messagebox.showwarning("Huy", "Chua chon thu muc input. Chuong trinh se ket thuc.")
        raise SystemExit

    print(">>> Hay chon folder OUTPUT (thu muc luu ket qua)...")
    OUTPUT_DIR = filedialog.askdirectory(title="Chon thu muc luu ket qua (output)")
    if not OUTPUT_DIR:
        messagebox.showwarning("Huy", "Chua chon thu muc output. Chuong trinh se ket thuc.")
        raise SystemExit

    process_batch_dir(INPUT_NORMAL_DIR, MARKER_X, MARKER_PLUS, OUTPUT_DIR)
