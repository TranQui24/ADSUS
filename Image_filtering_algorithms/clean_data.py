import os
from PIL import Image
import imagehash
from shutil import move

image_dir = r"D:\save_dataset\notinfected" 
trash_dir = r"D:\save_dataset\remove_image" 
os.makedirs(trash_dir, exist_ok=True)

MIN_RESOLUTION = 416  
HASH_THRESHOLD = 10   

saved_hashes = {}
deleted_by_res = 0
deleted_by_dup = 0

print("Bắt đầu quét và lọc dữ liệu...")

for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        continue
    img_path = os.path.join(image_dir, filename)
    should_move = False
    reason = ""
    
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            if width < MIN_RESOLUTION or height < MIN_RESOLUTION:
                reason = f"Kích thước quá nhỏ ({width}x{height})"
                should_move = True
            if not should_move:
                img_hash = imagehash.phash(img)
                # img_hash = imagehash.dhash(img)
                for old_filename, old_hash in saved_hashes.items():
                    if img_hash - old_hash <= HASH_THRESHOLD:
                        reason = f"Trùng cấu trúc video với {old_filename}"
                        should_move = True
                        break
                        
                if not should_move:
                    saved_hashes[filename] = img_hash
                    
    except Exception as e:
        print(f"[!] Lỗi khi xử lý file {filename}: {e}")
        continue 
    if should_move:
        try:
            print(f"[-] Loại bỏ {filename}: {reason}")
            move(img_path, os.path.join(trash_dir, filename))
            if "Kích thước" in reason:
                deleted_by_res += 1
            else:
                deleted_by_dup += 1
        except Exception as move_error:
            print(f"[!] Không thể di chuyển {filename}: {move_error}")

print("\n--- HOÀN THÀNH QUÁ TRÌNH LỌC ---")
print(f"Tổng số ảnh bị loại do độ phân giải thấp: {deleted_by_res}")
print(f"Tổng số ảnh bị loại do trùng lặp (cắt từ video): {deleted_by_dup}")
print(f"Số lượng ảnh sạch còn lại trong thư mục: {len(saved_hashes)}")