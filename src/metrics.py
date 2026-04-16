import numpy as np
from scipy.ndimage import distance_transform_edt


def compute_confusion_matrix(pred, target, num_classes=21, ignore_index=255):
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (target, pred), 1)
    return confusion


def pixel_accuracy(confusion):
    return np.diag(confusion).sum() / (confusion.sum() + 1e-10)


def per_class_iou(confusion):
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    return tp / (tp + fp + fn + 1e-10)


def per_class_accuracy(confusion):
    tp = np.diag(confusion)
    total = confusion.sum(axis=1)
    return tp / (total + 1e-10)


def per_class_dice(confusion):
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    return 2 * tp / (2 * tp + fp + fn + 1e-10)


def _erode(mask):
    padded = np.pad(mask, 1, mode='constant', constant_values=False)
    return (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] &
            padded[1:-1, :-2] & padded[1:-1, 2:])


def hausdorff_95(pred_mask, gt_mask):
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return np.nan

    pred_boundary = pred_mask & ~_erode(pred_mask)
    gt_boundary = gt_mask & ~_erode(gt_mask)

    if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
        return np.nan

    dt_gt = distance_transform_edt(~gt_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)

    d_pred_to_gt = dt_gt[pred_boundary]
    d_gt_to_pred = dt_pred[gt_boundary]

    return max(np.percentile(d_pred_to_gt, 95), np.percentile(d_gt_to_pred, 95))


def compute_hd95_per_class(pred, target, num_classes=21, ignore_index=255):
    results = {}
    valid = target != ignore_index
    for c in range(num_classes):
        pred_c = (pred == c) & valid
        gt_c = (target == c) & valid
        if gt_c.sum() > 0:
            results[c] = hausdorff_95(pred_c, gt_c)
    return results


def compute_all_metrics(all_preds, all_targets, num_classes=21, ignore_index=255):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, target in zip(all_preds, all_targets):
        confusion += compute_confusion_matrix(
            pred[np.newaxis], target[np.newaxis], num_classes, ignore_index
        )

    iou = per_class_iou(confusion)
    dice = per_class_dice(confusion)
    acc = per_class_accuracy(confusion)

    hd95_per_class = {c: [] for c in range(num_classes)}
    for pred, target in zip(all_preds, all_targets):
        hd95_img = compute_hd95_per_class(pred, target, num_classes, ignore_index)
        for c, val in hd95_img.items():
            if not np.isnan(val):
                hd95_per_class[c].append(val)

    mean_hd95 = {}
    for c in range(num_classes):
        if hd95_per_class[c]:
            mean_hd95[c] = np.mean(hd95_per_class[c])

    present = confusion.sum(axis=1) > 0

    return {
        "pixel_accuracy": pixel_accuracy(confusion),
        "mean_iou": np.mean(iou[present]),
        "mean_dice": np.mean(dice[present]),
        "per_class_iou": iou,
        "per_class_dice": dice,
        "per_class_accuracy": acc,
        "mean_hd95": np.nanmean(list(mean_hd95.values())) if mean_hd95 else np.nan,
        "per_class_hd95": mean_hd95,
        "confusion_matrix": confusion,
    }
