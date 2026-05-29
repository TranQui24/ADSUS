import os
import cv2
import numpy as np

def create_side_by_side_comparison(gt_dir, pred_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    valid_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in os.listdir(gt_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

    count = 0
    for img_name in image_files:
        gt_path = os.path.join(gt_dir, img_name)
        pred_path = os.path.join(pred_dir, img_name)
        if not os.path.exists(pred_path):
            print(f"[-] Bo qua: Khong tim thay anh du doan cho {img_name}")
            continue

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
        cv2.putText(header_pred, "AI PREDICTION (YOLOv8)", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        img_gt_with_header = cv2.vconcat([header_gt, img_gt])
        img_pred_with_header = cv2.vconcat([header_pred, img_pred])
            

        separator = np.full((h_gt + header_height, 5, 3), 255, dtype=np.uint8)
        combined_img = cv2.hconcat([img_gt_with_header, separator, img_pred_with_header])

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, combined_img)
        count += 1
        
    print(f"[+] Hoan tat! Da ghep song song {count} cap anh.")
    print(f"[+] Thu muc ket qua: {output_dir}")

if __name__ == '__main__':

    GT_DIR = r"D:\AI_Data\test_visualized_ground_truth" 
    

    PRED_DIR = r"D:\AI_Data\inference_results\test_87_images" 

    OUTPUT_DIR = r"D:\AI_Data\test_comparison_side_by_side" 
    
    create_side_by_side_comparison(GT_DIR, PRED_DIR, OUTPUT_DIR)