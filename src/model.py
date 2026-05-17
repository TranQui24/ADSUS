import torch
from ultralytics import YOLO

def get_hybrid_yolo(model_name="yolov8n.pt", in_channels=5, num_classes=1):
    model = YOLO(model_name)
    
    if in_channels != 3:
        old_conv = model.model.model[0].conv
        new_conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            torch.nn.init.kaiming_normal_(new_conv.weight[:, 3:], mode='fan_out', nonlinearity='relu')
            
        model.model.model[0].conv = new_conv
        
    return model