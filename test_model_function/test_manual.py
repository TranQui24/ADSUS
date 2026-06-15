import os
import cv2
import numpy as np
import shutil
from ultralytics import YOLO

def run_manual_inference(model_path, source_dir, project_dir, name):
    model = YOLO(model_path)
    model.predict(
        source=source_dir,
        save=True,       
        save_txt=True,     
        conf=0.25,          
        project=project_dir, 
        name=name                    
    )
    return os.path.join(project_dir, name)

def create_side_by_side_comparison(gt_dir, pred_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ['.jpg', '.jpeg', '.png']
    
    gt_files = set(f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in valid_extensions)
    pred_files = set(f for f in os.listdir(pred_dir) if os.path.splitext(f)[1].lower() in valid_extensions)
    
    all_files = gt_files.union(pred_files)

    count = 0
    for img_name in all_files:
        gt_path = os.path.join(gt_dir, img_name)
        pred_path = os.path.join(pred_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        
        has_gt = os.path.exists(gt_path)
        has_pred = os.path.exists(pred_path)

        if has_gt and has_pred:
            img_gt = cv2.imread(gt_path)
            img_pred = cv2.imread(pred_path)
            
            if img_gt is None or img_pred is None:
                continue

            h_gt, w_gt = img_gt.shape[:2]
            h_pred, w_pred = img_pred.shape[:2]
            
            if h_gt != h_pred or w_gt != w_pred:
                img_pred = cv2.resize(img_pred, (w_gt, h_gt))

            header_height = 50
            header_gt = np.zeros((header_height, w_gt, 3), dtype=np.uint8)
            header_pred = np.zeros((header_height, w_gt, 3), dtype=np.uint8)

            cv2.putText(header_gt, "GROUND TRUTH (Bac si khoanh)", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(header_pred, "AI PREDICTION (YOLO)", (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            img_gt_with_header = cv2.vconcat([header_gt, img_gt])
            img_pred_with_header = cv2.vconcat([header_pred, img_pred])
                
            separator = np.full((h_gt + header_height, 5, 3), 255, dtype=np.uint8)
            combined_img = cv2.hconcat([img_gt_with_header, separator, img_pred_with_header])

            cv2.imwrite(output_path, combined_img)
            count += 1
            
        elif has_pred and not has_gt:
            shutil.copy2(pred_path, output_path)
            count += 1
            
        elif has_gt and not has_pred:
            shutil.copy2(gt_path, output_path)
            count += 1
            
    print(f"[+] Hoan tat! Da xu ly {count} anh.")
    print(f"[+] Thu muc ket qua: {output_dir}")

if __name__ == '__main__':
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox

    # Khoi tao cua so tkinter an di
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    # --- Model path giu nguyen, it thay doi ---
    MODEL_PATH = r"D:\AI_Data\training_runs\YoLov11_Small_416_Augmented2\weights\best.pt"

    # --- Chon folder anh test ---
    print("[?] Chon thu muc anh TEST...")
    TEST_IMAGES_DIR = filedialog.askdirectory(title="Chon thu muc anh TEST (TEST_IMAGES_DIR)")
    if not TEST_IMAGES_DIR:
        messagebox.showerror("Loi", "Chua chon thu muc anh test. Thoat.")
        exit()

    # --- Chon PROJECT folder (noi luu ket qua predict) ---
    print("[?] Chon thu muc PROJECT (noi YOLO luu ket qua predict)...")
    PROJECT_DIR = filedialog.askdirectory(title="Chon thu muc PROJECT (PROJECT_DIR)")
    if not PROJECT_DIR:
        messagebox.showerror("Loi", "Chua chon thu muc project. Thoat.")
        exit()

    # --- Nhap ten sub-folder ket qua predict ---
    PREDICT_NAME = simpledialog.askstring(
        title="Ten ket qua predict",
        prompt="Nhap ten thu muc con luu ket qua (PREDICT_NAME):",
        initialvalue="predict_result",
        parent=root
    )
    if not PREDICT_NAME:
        messagebox.showerror("Loi", "Chua nhap ten ket qua. Thoat.")
        exit()

    # --- Ground Truth co dinh, khong thay doi ---
    GT_DIR = r"D:\AI_Data\test_visualized_ground_truth"

    # --- Chon folder OUTPUT cuoi cung ---
    print("[?] Chon thu muc OUTPUT (luu anh so sanh cuoi cung)...")
    OUTPUT_DIR = filedialog.askdirectory(title="Chon thu muc OUTPUT ket qua cuoi (OUTPUT_DIR)")
    if not OUTPUT_DIR:
        messagebox.showerror("Loi", "Chua chon thu muc output. Thoat.")
        exit()

    root.destroy()

    # --- In lai cau hinh da chon ---
    print("\n[*] Cau hinh da chon:")
    print(f"    MODEL_PATH    : {MODEL_PATH}")
    print(f"    TEST_IMAGES   : {TEST_IMAGES_DIR}")
    print(f"    PROJECT_DIR   : {PROJECT_DIR}")
    print(f"    PREDICT_NAME  : {PREDICT_NAME}")
    print(f"    GT_DIR        : {GT_DIR}")
    print(f"    OUTPUT_DIR    : {OUTPUT_DIR}\n")

    print("[*] Bat dau chay suy luan mo hinh...")
    pred_dir = run_manual_inference(MODEL_PATH, TEST_IMAGES_DIR, PROJECT_DIR, PREDICT_NAME)

    print("[*] Bat dau ghep anh so sanh...")
    create_side_by_side_comparison(GT_DIR, pred_dir, OUTPUT_DIR)