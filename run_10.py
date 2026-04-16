#!/usr/bin/env python3
"""Run the 10 segmentation experiments at 512x512 and log to autoresearch_log.tsv."""

import csv
import os
import subprocess
import sys
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(PROJECT_DIR, "autoresearch_log.tsv")
IMG_SIZE = 512

EXPERIMENTS = [
    {
        "name": "1_unet_r18_baseline",
        "train": "--model unet --backbone resnet18 --loss combined --augment --normalize --epochs 150 --batch-size 8 --lr 1e-4",
        "eval": "--model unet --backbone resnet18 --normalize",
        "ckpt": "unet_resnet18_combined_aug_norm",
    },
    {
        "name": "2_unet_r50",
        "train": "--model unet --backbone resnet50 --loss combined --augment --normalize --epochs 150 --batch-size 8 --lr 1e-4",
        "eval": "--model unet --backbone resnet50 --normalize",
        "ckpt": "unet_resnet50_combined_aug_norm",
    },
    {
        "name": "3_unet_r18_noaug",
        "train": "--model unet --backbone resnet18 --loss combined --normalize --epochs 150 --batch-size 8 --lr 1e-4",
        "eval": "--model unet --backbone resnet18 --normalize",
        "ckpt": "unet_resnet18_combined_norm",
    },
    {
        "name": "4_unet_r18_ceonly",
        "train": "--model unet --backbone resnet18 --loss ce --augment --normalize --epochs 150 --batch-size 8 --lr 1e-4",
        "eval": "--model unet --backbone resnet18 --normalize",
        "ckpt": "unet_resnet18_ce_aug_norm",
    },
    {
        "name": "5_dlv3p_os16",
        "train": "--model deeplabv3plus --backbone resnet50 --output-stride 16 --loss combined --augment --normalize --epochs 75 --batch-size 8 --lr 5e-5",
        "eval": "--model deeplabv3plus --backbone resnet50 --output-stride 16 --normalize",
        "ckpt": "deeplabv3plus_resnet50_os16_combined_aug_norm",
    },
    {
        "name": "6_dlv3p_os8",
        "train": "--model deeplabv3plus --backbone resnet50 --output-stride 8 --loss combined --augment --normalize --epochs 75 --batch-size 8 --lr 5e-5",
        "eval": "--model deeplabv3plus --backbone resnet50 --output-stride 8 --normalize",
        "ckpt": "deeplabv3plus_resnet50_os8_combined_aug_norm",
    },
    {
        "name": "7_dlv3p_os16_noaug",
        "train": "--model deeplabv3plus --backbone resnet50 --output-stride 16 --loss combined --normalize --epochs 75 --batch-size 8 --lr 5e-5",
        "eval": "--model deeplabv3plus --backbone resnet50 --output-stride 16 --normalize",
        "ckpt": "deeplabv3plus_resnet50_os16_combined_norm",
    },
    {
        "name": "8_sam_large_finetuned",
        "train": "--model sam --sam-size large --loss combined --augment --normalize --epochs 75 --batch-size 4 --lr 1e-4",
        "eval": "--model sam --sam-size large --normalize",
        "ckpt": "sam_large_combined_aug_norm",
    },
    {
        "name": "9_sam_large_frozen",
        "train": "--model sam --sam-size large --loss combined --augment --normalize --epochs 75 --batch-size 4 --lr 1e-3 --freeze-encoder",
        "eval": "--model sam --sam-size large --freeze-encoder --normalize",
        "ckpt": "sam_large_combined_frozen_aug_norm",
    },
    {
        "name": "10_sam_large_frozen_ce",
        "train": "--model sam --sam-size large --loss ce --augment --normalize --epochs 75 --batch-size 4 --lr 1e-3 --freeze-encoder",
        "eval": "--model sam --sam-size large --freeze-encoder --normalize",
        "ckpt": "sam_large_ce_frozen_aug_norm",
    },
]


def log_result(name, phase, status, elapsed, notes=""):
    exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not exists:
            writer.writerow(["timestamp", "experiment", "phase", "status", "elapsed_s", "notes"])
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            name, phase, status, f"{elapsed:.0f}", notes,
        ])


def run_cmd(cmd_str, name, phase):
    print(f"\n{'='*70}")
    print(f"[{phase.upper()}] {name}")
    print(f"  CMD: {cmd_str}")
    print(f"{'='*70}")
    t0 = time.time()
    result = subprocess.run(cmd_str, shell=True, cwd=PROJECT_DIR)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    log_result(name, phase, status, elapsed)
    print(f"  [{status}] {elapsed:.0f}s")
    return result.returncode == 0


def main():
    python = sys.executable
    data_root = "./dataset"

    for exp in EXPERIMENTS:
        name = exp["name"]
        ckpt_dir = f"checkpoints/{exp['ckpt']}"

        best_path = os.path.join(PROJECT_DIR, ckpt_dir, "best.pth")
        if os.path.exists(best_path):
            print(f"\n>>> Skipping training for {name} (checkpoint exists: {ckpt_dir})")
            log_result(name, "train", "SKIPPED", 0, "checkpoint exists")
        else:
            train_cmd = (f"{python} scripts/train.py {exp['train']} "
                         f"--data-root {data_root} --img-size {IMG_SIZE}")
            ok = run_cmd(train_cmd, name, "train")
            if not ok:
                print(f"  !!! Training failed for {name}, skipping eval")
                continue

        eval_cmd = (f"{python} scripts/evaluate.py {exp['eval']} "
                    f"--checkpoint-dir {ckpt_dir} --data-root {data_root} "
                    f"--img-size {IMG_SIZE}")
        run_cmd(eval_cmd, name, "eval")


if __name__ == "__main__":
    main()
