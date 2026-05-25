import cv2
import os
from shutil import move
from skimage.metrics import structural_similarity as ssim

image_dir = r"D:\save_dataset\notinfected"
trash_dir = r"D:\save_dataset\remove_image"

SSIM_THRESHOLD = 0.88 
TARGET_SIZE = (416, 416) 

saved_images = {} 
deleted_count = 0

print("Bắt đầu lọc trùng lặp bằng thuật toán cấu trúc SSIM...", flush=True)

all_files = [f for f in sorted(os.listdir(image_dir)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
total_files = len(all_files)
print(f"Tìm thấy tổng cộng {total_files} file ảnh cần xử lý.\n", flush=True)

for idx, filename in enumerate(all_files, start=1):
    img_path = os.path.join(image_dir, filename)
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
        
    img_resized = cv2.resize(img, TARGET_SIZE)
    is_duplicate = False
    
    for old_filename, old_img in saved_images.items():
        score = ssim(old_img, img_resized)
        
        if score > SSIM_THRESHOLD:
            # Thêm flush=True ở đây
            print(f"[{idx}/{total_files}] [-] Xóa: {filename} trùng cấu trúc với {old_filename} (SSIM: {score:.4f})", flush=True)
            move(img_path, os.path.join(trash_dir, filename))
            is_duplicate = True
            deleted_count += 1
            break
            
    if not is_duplicate:
        saved_images[filename] = img_resized
        if idx % 10 == 0 or idx == total_files:
            # Thêm flush=True ở đây
            print(f"[{idx}/{total_files}] Đang xử lý... Đang giữ lại {len(saved_images)} ảnh sạch.", flush=True)

print(f"\n--- HOÀN THÀNH ---", flush=True)
print(f"Số lượng ảnh trùng lặp bị loại bỏ: {deleted_count}", flush=True)
print(f"Số lượng ảnh sạch giữ lại: {len(saved_images)}", flush=True)