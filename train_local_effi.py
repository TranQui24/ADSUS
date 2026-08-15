import os
import numpy as np
from ultralytics import settings, YOLO
from ultralytics.utils.metrics import DetMetrics

os.environ["WANDB_ENTITY"] = "set-g64"
os.environ["WANDB_PROJECT"] = "ADSUS"

# 1. Custom fitness (giữ nguyên từ bản gốc)
def custom_fitness(self):

    w = np.array([0.0, 0.6, 0.4, 0.0])
    results = np.array(self.mean_results())
    return (results * w).sum()

DetMetrics.fitness = property(custom_fitness)


def main():
    data_base_dir = r"D:\AI_Data"
    settings.update({"wandb": True})

    model = YOLO("yolo26-effnet.yaml")

    backbone_layer = model.model.model[0] 
    frozen_params = 0
    for i, sub in enumerate(backbone_layer.m):
        if i < 4:
            for p in sub.parameters():
                p.requires_grad = False
                frozen_params += p.numel()
    print(f"Frozen {frozen_params:,} params in backbone stages 0-3")

    model.train(
        data="data.yaml",
        project=os.path.join(data_base_dir, "ADSUS"),
        name="YOLO26_EffNetV2S_512_40H1_nbl",
        epochs=200,
        batch=4,
        nbs=16,          
        imgsz=512,       
        device=0,
        workers=2,
        optimizer='AdamW',
        lr0=1e-4,
        amp=True,         
        patience=50,
        cls=1.0,
        conf=0.15,
        weight_decay=0.001,
        label_smoothing=0.1,     
        degrees=15.0,
        flipud=0.5,
        fliplr=0.5,
        scale=0.5,
        perspective=0.0001,
        mosaic=1.0,
        hsv_v=0.4,
        erasing=0.1,
    )


if __name__ == '__main__':
    main()