# File: efficientnet_backbone.py (tạm đặt ở thư mục gốc project để test trước khi đưa vào ultralytics/nn/modules/)
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class EfficientNetV2SBackbone(nn.Module):
    """
    EfficientNetV2-S wrapped as a YOLO-compatible backbone.
    Outputs 3 feature maps at strides 8/16/32 (P3, P4, P5)
    to feed directly into YOLO's Neck (PANet).
    """
    def __init__(self, pretrained=True, freeze_stages=4):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base = efficientnet_v2_s(weights=weights)
        self.features = base.features  # nn.Sequential, 8 stages (index 0-7)

        # torchvision EfficientNetV2-S stage map (verified for torchvision 0.20.1):
        # idx 0: stem            -> stride 2,  24ch
        # idx 1: stage1 (fused)  -> stride 4,  24ch
        # idx 2: stage2 (fused)  -> stride 8,  48ch   <- P3
        # idx 3: stage3 (fused)  -> stride 16, 64ch
        # idx 4: stage4 (MBConv) -> stride 16, 128ch
        # idx 5: stage5 (MBConv) -> stride 16, 160ch  <- P4
        # idx 6: stage6 (MBConv) -> stride 32, 256ch  <- P5
        # idx 7: final 1x1 conv  -> 1280ch (dropped, classification-only)
        self.p3_idx = 3
        self.p4_idx = 5
        self.p5_idx = 6
        self.out_channels = [48, 160, 256]  # dùng cho bước sửa YAML Neck sau

        if freeze_stages > 0:
            for i, layer in enumerate(self.features):
                if i <= freeze_stages:
                    for p in layer.parameters():
                        p.requires_grad = False

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features[:self.p5_idx + 1]):
            x = layer(x)
            if i in (self.p3_idx, self.p4_idx, self.p5_idx):
                feats.append(x)
        return feats  # [P3, P4, P5]


if __name__ == "__main__":
    # Test nhanh ngay trong file này — chạy: python efficientnet_backbone.py
    model = EfficientNetV2SBackbone(pretrained=True, freeze_stages=4)
    model.eval()
    x = torch.randn(1, 3, 448, 448)
    with torch.no_grad():
        p3, p4, p5 = model(x)
    print("P3:", p3.shape)
    print("P4:", p4.shape)
    print("P5:", p5.shape)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")