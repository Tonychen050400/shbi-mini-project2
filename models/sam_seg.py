import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMSegHead(nn.Module):
    def __init__(self, in_channels=256, num_classes=21):
        super().__init__()
        self.conv_high = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_mid = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv_low = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(128 * 3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, fpn_features):
        target_size = fpn_features[0].shape[2:]

        f_high = self.conv_high(fpn_features[0])
        f_mid = F.interpolate(self.conv_mid(fpn_features[1]), size=target_size,
                              mode="bilinear", align_corners=False)
        f_low = F.interpolate(self.conv_low(fpn_features[2]), size=target_size,
                              mode="bilinear", align_corners=False)

        fused = torch.cat([f_high, f_mid, f_low], dim=1)
        return self.fuse(fused)


class SAMSeg(nn.Module):
    def __init__(self, sam2_model, num_classes=21, freeze_encoder=True):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False
        self.seg_head = SAMSegHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        input_size = x.shape[2:]
        if self.freeze_encoder:
            with torch.no_grad():
                features = self.image_encoder(x)
                fpn = features["backbone_fpn"]
        else:
            features = self.image_encoder(x)
            fpn = features["backbone_fpn"]

        out = self.seg_head(fpn)
        return F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)


SAM2_CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
}


def build_sam_seg(size="tiny", num_classes=21, freeze_encoder=True, device="cuda"):
    from sam2.build_sam import build_sam2
    if size not in SAM2_CONFIGS:
        raise ValueError(f"Unknown SAM2 size: {size}. Choose from {list(SAM2_CONFIGS.keys())}")
    model_cfg, checkpoint_path = SAM2_CONFIGS[size]
    sam2_model = build_sam2(model_cfg, checkpoint_path, device=device)
    model = SAMSeg(sam2_model, num_classes=num_classes, freeze_encoder=freeze_encoder)
    return model.to(device)
