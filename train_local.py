import os
from ultralytics import settings, YOLO

os.environ["WANDB_ENTITY"] = "set-g64"
os.environ["WANDB_PROJECT"] = "ADSUS" 

def main():
    data_base_dir = r"D:\AI_Data"

    settings.update({"wandb": True})

    model = YOLO("yolo26s.pt")

    model.train(
        data="data.yaml",
        project=os.path.join(data_base_dir, "ADSUS"),
        name="YOLO26s_512_Execution1",
        epochs=200,                          # Tăng từ 150 → 200 để bù batch nhỏ hơn
        batch=4,                             # Giảm từ 8 → 4 để tránh OOM với imgsz=512
        nbs=8,                               # Nominal batch size: scale lr như batch=8
        imgsz=512,                           # Tăng từ 416 → 512 (bội số 32, chi tiết hơn)
        device=0,
        workers=2,
        optimizer='AdamW',
        lr0=1e-4,
        patience=30,                         # Tăng từ 20 → 30 tương ứng epochs nhiều hơn

        # --- Regularization (thay thế dropout, phù hợp dataset nhỏ 800 ảnh) ---
        weight_decay=0.001,                  # L2 regularization, chống overfit
        label_smoothing=0.1,                 # Giảm overfit nhãn cứng
        freeze=10,                           # Đóng băng 10 layer đầu backbone pretrained

        # --- Augmentation (chỉ giữ kỹ thuật phù hợp với ảnh siêu âm grayscale) ---
        degrees=15.0,                        # Rotation nhỏ, mô phỏng thay góc đầu dò
        flipud=0.5,                          # Lật dọc
        fliplr=0.5,                          # Lật ngang
        scale=0.5,                           # Zoom in/out, mô phỏng thay đổi depth
        perspective=0.0001,                  # Biến dạng phối cảnh nhỏ
        mosaic=1.0,                          # Ghép 4 ảnh, tăng đa dạng context
        hsv_v=0.4,                           # Thay đổi brightness → mô phỏng gain máy siêu âm
        erasing=0.1,                         # Random erasing → mô phỏng acoustic shadow
    )

if __name__ == '__main__':
    main()
