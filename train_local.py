import os
import xml.etree.ElementTree as ET
from ultralytics import settings, YOLO

os.environ["WANDB_ENTITY"] = "set-g64"

def main():
    data_base_dir = r"D:\AI_Data"

    settings.update({"wandb": True})

    model = YOLO("yolo11s.pt")

    model.train(
        data="data.yaml",                    
        project=os.path.join(data_base_dir, "AI_SYSTEM_FOR_UTERINE_ULTRASOUND_IMAGE_ANALYSIS_AND_ABNORMALITY_DETECTION"),  
        name="YOLOv11s_416_Augmented_Execution3",
        epochs=150,                          
        batch=8,                             
        imgsz=416,                           
        device=0,                            
        workers=2,                           
        optimizer='AdamW',
        lr0=1e-4,
        patience=20,
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        perspective=0.0001,
        mosaic=1.0
    )

if __name__ == '__main__':
    main()