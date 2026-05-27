import os
image_dir = r"D:\save_dataset\notinfected"

valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

print("Bắt đầu khởi tạo tệp nhãn .txt rỗng...", flush=True)

success_count = 0
exist_count = 0
for filename in os.listdir(image_dir):
    if filename.lower().endswith(valid_extensions):
        base_name = os.path.splitext(filename)[0]

        txt_filename = f"{base_name}.txt"
        txt_path = os.path.join(image_dir, txt_filename)

        if not os.path.exists(txt_path):
            with open(txt_path, 'w') as f:
                pass  
            success_count += 1
        else:
            exist_count += 1

print("\n--- HOÀN THÀNH QUÁ TRÌNH KHỞI TẠO ---", flush=True)
print(f"Số lượng file .txt rỗng đã tạo mới: {success_count}", flush=True)
print(f"Số lượng file .txt đã tồn tại từ trước: {exist_count}", flush=True)