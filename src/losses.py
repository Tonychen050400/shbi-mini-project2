import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, smooth=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        valid = targets != self.ignore_index
        targets_clean = targets.clone()
        targets_clean[~valid] = 0

        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets_clean, self.num_classes).permute(0, 3, 1, 2).float()

        valid_mask = valid.unsqueeze(1).float()
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        cardinality = (probs + one_hot).sum(dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=21, ignore_index=255, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
