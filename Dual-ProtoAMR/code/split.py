#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
from collections import defaultdict

from config import get_cfg

# ======================
# 超参
# ======================
SEED = 42
RATIO = 0.20  # 20% labeled, 80% unlabeled
SAVE_SUBSETS = True


# ======================
# 数据工具
# ======================
def unify_x(x, seq_len):
    # 确保输入形状为 (N, seq_len, 2)
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got {x.shape}")
    if x.shape[1] == seq_len and x.shape[2] == 2:
        return x
    if x.shape[1] == 2 and x.shape[2] == seq_len:
        return np.transpose(x, (0, 2, 1))  # 转置为 (N, seq_len, 2)
    raise ValueError(f"Unrecognized X shape: {x.shape}, expected seq_len={seq_len}")


def unify_y(y, n):
    # 确保标签形状为 (N, 2)
    if y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {y.shape}")
    if y.shape == (n, 2):
        return y
    if y.shape == (2, n):
        return y.T
    raise ValueError(f"Unrecognized Y shape: {y.shape}")


# ======================
# 分层划分（label, snr）
# ======================
def stratified_split_by_label_snr(labels, snrs, ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n_total = len(labels)

    buckets = defaultdict(list)
    for idx in range(n_total):
        buckets[(int(labels[idx]), int(snrs[idx]))].append(idx)

    labeled_idx = []
    unlabeled_idx = []
    for _, idxs in buckets.items():
        idxs = np.array(idxs, dtype=np.int32)
        rng.shuffle(idxs)

        n = len(idxs)
        n_l = int(np.round(n * ratio))
        if n >= 2:
            n_l = max(1, min(n - 1, n_l))
        else:
            n_l = 1

        labeled_idx.append(idxs[:n_l])
        unlabeled_idx.append(idxs[n_l:])

    labeled_idx = np.concatenate(labeled_idx).astype(np.int32)
    unlabeled_idx = np.concatenate(unlabeled_idx).astype(np.int32)

    rng.shuffle(labeled_idx)
    rng.shuffle(unlabeled_idx)
    return labeled_idx, unlabeled_idx


# ======================
# 主流程
# ======================
def main(dataset_name="radio128"):
    cfg = get_cfg(dataset_name)
    x_path = cfg["x_path"]
    y_path = cfg["y_path"]
    seq_len = cfg["seq_len"]
    out_dir = cfg["data_dir"]

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading data for {dataset_name} ...")
    x = np.load(x_path)
    y = np.load(y_path)

    x = unify_x(x, seq_len)
    y = unify_y(y, x.shape[0]).astype(np.int32)

    labels = y[:, 0]
    snrs = y[:, 1]
    n_total = x.shape[0]

    labeled_idx, unlabeled_idx = stratified_split_by_label_snr(labels, snrs, ratio=RATIO, seed=SEED)

    np.save(os.path.join(out_dir, "idx_labeled_20.npy"), labeled_idx)
    np.save(os.path.join(out_dir, "idx_unlabeled_80.npy"), unlabeled_idx)

    if SAVE_SUBSETS:
        x_l, y_l = x[labeled_idx], y[labeled_idx]
        x_u, y_u = x[unlabeled_idx], y[unlabeled_idx]

        np.save(os.path.join(out_dir, "X_labeled_20.npy"), x_l.astype(np.float32))
        np.save(os.path.join(out_dir, "Y_labeled_20.npy"), y_l.astype(np.int16))
        np.save(os.path.join(out_dir, "X_unlabeled_80.npy"), x_u.astype(np.float32))
        np.save(os.path.join(out_dir, "Y_unlabeled_80.npy"), y_u.astype(np.int16))

    meta = {
        "dataset_name": dataset_name,
        "x_path": x_path,
        "y_path": y_path,
        "seq_len": int(seq_len),
        "seed": int(SEED),
        "ratio_labeled": float(RATIO),
        "N_total": int(n_total),
        "N_labeled": int(len(labeled_idx)),
        "N_unlabeled": int(len(unlabeled_idx)),
        "unique_labels": int(len(np.unique(labels))),
        "unique_snrs": int(len(np.unique(snrs))),
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("========== SPLIT DONE ==========")
    print(meta)
    print("Saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alldb_radio512", choices=["radio128", "radio512","radio1024"])
    args = parser.parse_args()
    main(args.dataset)
