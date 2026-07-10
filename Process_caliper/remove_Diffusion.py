import os
import torch
from PIL import Image, ImageOps, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

# 1. Cấu hình đường dẫn thư mục
IMG_DIR = "images"
MASK_DIR = "masks"
OUT_DIR = "clean_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("--- Đang tải mô hình Stable Diffusion v1-5 (Thuật toán Hoà Trộn Viền Cục Bộ) ---")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting", 
    torch_dtype=torch.float32
)

pipe.safety_checker = None
pipe.requires_safety_checker = False

# Tối ưu hóa VRAM tối đa cho card 4GB
pipe.enable_attention_slicing("max")  
pipe.enable_sequential_cpu_offload() 

# 4. Quét danh sách file ảnh gốc
files = [f for f in os.listdir(IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"Tìm thấy {len(files)} ảnh cần xử lý.")

# 5. Vòng lặp xử lý thông minh bảo toàn ảnh gốc + mịn vùng vá
for idx, file in enumerate(files):
    filename_without_ext = os.path.splitext(file)[0]
    
    img_path = os.path.join(IMG_DIR, file)
    out_path = os.path.join(OUT_DIR, file)
    
    mask_path_png = os.path.join(MASK_DIR, f"{filename_without_ext}.png")
    mask_path_jpg = os.path.join(MASK_DIR, f"{filename_without_ext}.jpg")
    
    if os.path.exists(mask_path_png):
        mask_path = mask_path_png
    elif os.path.exists(mask_path_jpg):
        mask_path = mask_path_jpg
    else:
        print(f"Bỏ qua {file} vì không tìm thấy mask tương ứng.")
        continue
        
    print(f"[{idx+1}/{len(files)}] Đang xử lý hoà trộn mịn cục bộ: {file}...")
    
    # Đọc ảnh gốc để giữ nguyên chất lượng nền xung quanh
    original_image = Image.open(img_path).convert("RGB")
    w_orig, h_orig = original_image.size
    mask_orig = Image.open(mask_path).convert("L")
    
    # 🛠️ BƯỚC 1: Nới rộng nhẹ mask gốc để AI vẽ loang ra, tạo vùng đệm mịn
    mask_ai_medium = mask_orig.filter(ImageFilter.MaxFilter(size=9))
    
    # Kích thước chia hết cho 8 cho Stable Diffusion
    new_w = (w_orig // 8) * 8
    new_h = (h_orig // 8) * 8
    
    init_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    mask_for_ai = mask_ai_medium.resize((new_w, new_h), Image.Resampling.NEAREST)
    
    # Cho Stable Diffusion vẽ vùng cần xóa trên mask đệm
    with torch.inference_mode():
        ai_output = pipe(
            prompt="ultrasound tissue texture, seamless match", 
            image=init_image, 
            mask_image=mask_for_ai, 
            width=new_w, 
            height=new_h, 
            num_inference_steps=20
        ).images[0]
        
    # Đưa ảnh AI vẽ về đúng kích thước gốc ban đầu
    ai_output = ai_output.resize((w_orig, h_orig), Image.Resampling.LANCZOS)
    
    # KHỬ VÀNG CỤC BỘ: Chỉ biến riêng vùng AI tạo ra thành Grayscale để đồng bộ màu xám
    ai_output_gray = ImageOps.grayscale(ai_output).convert("RGB")
    
    # 🛠️ BƯỚC 2: Tạo mask trộn rộng hơn vùng vẽ một chút và làm mờ viền nhẹ nhàng
    mask_large = mask_orig.filter(ImageFilter.MaxFilter(size=15))
    mask_blend = mask_large.filter(ImageFilter.GaussianBlur(radius=5)) # Radius=5 là vừa đủ mịn, không bị loang quá đà
    
    # Tiến hành hoà trộn: Lấy mẩu thịt AI dán đè lên ảnh gốc thông qua mask viền mờ cục bộ
    final_combined = Image.composite(ai_output_gray, original_image, mask_blend)
    
    # Lưu lại thành quả
    final_combined.save(out_path)

print("--- complete! ---")