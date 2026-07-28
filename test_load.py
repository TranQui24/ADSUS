from ultralytics import YOLO
import torch

model = YOLO(r"D:\Code_for_set\ADSUS\env_yolo\Lib\site-packages\ultralytics\cfg\models\26\yolo26-effnet.yaml")
x = torch.randn(1, 3, 448, 448)
model.model.eval()
with torch.no_grad():
    out = model.model(x)
print("Output type:", type(out))
if isinstance(out, (list, tuple)):
    for o in out:
        print("shape:", o.shape if hasattr(o, "shape") else type(o))