#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合 IQ 域(CNN2D embedding) + 图像域(CLIP embedding) 的“无训练”自适应加权近邻判决
- 不训练 CNN / 不训练 CLIP / 不训练新分类器
- 用少量有标签(20%)构建两域的类原型库
- 对无标签(80%)计算两域相似度向量、置信度(margin)、softmax 得权重
- 融合相似度向量后输出最终类别，并统计融合后精度/按SNR精度
"""

import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from config import get_cfg, DATASET_CONFIG

DATASET_CONFIG_KEYS = sorted(DATASET_CONFIG.keys())

# ======================
# 超参
# ======================
TEMPERATURE = 1
USE_REJECT = False
REJECT_MAXSIM_TH = 0.0
REJECT_MARGIN_TH = 0.0
REJECT_LABEL = -1


# ======================
# 工具函数
# ======================
def l2norm(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def softmax2(a: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = a / max(T, 1e-12)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def load_embeddings(img_dir, iq_dir):
    idx_l = np.load(os.path.join(img_dir, "idx_labeled_20.npy"))
    idx_u = np.load(os.path.join(img_dir, "idx_unlabeled_80.npy"))

    f_l_img = np.load(os.path.join(img_dir, "F_labeled.npy")).astype(np.float32)
    f_u_img = np.load(os.path.join(img_dir, "F_unlabeled.npy")).astype(np.float32)

    f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy")).astype(np.float32)
    f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy")).astype(np.float32)

    f_l_img = l2norm(f_l_img)
    f_u_img = l2norm(f_u_img)
    f_l_iq = l2norm(f_l_iq)
    f_u_iq = l2norm(f_u_iq)

    if f_l_img.shape[0] != f_l_iq.shape[0]:
        raise ValueError(
            f"labeled size mismatch: img {f_l_img.shape} vs iq {f_l_iq.shape}. "
            f"请确保 run_iq.py 使用了同一 idx_labeled_20.npy 切分并保持顺序。"
        )
    if f_u_img.shape[0] != f_u_iq.shape[0]:
        raise ValueError(
            f"unlabeled size mismatch: img {f_u_img.shape} vs iq {f_u_iq.shape}. "
            f"请确保 run_iq.py 使用了同一 idx_unlabeled_80.npy 切分并保持顺序。"
        )

    return idx_l, idx_u, f_l_img, f_u_img, f_l_iq, f_u_iq


def load_labels(y_path, idx_l, idx_u):
    y = np.load(y_path)
    if y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {y.shape}")
    if y.shape[0] == 2:
        y = y.T
    y = y.astype(np.int32)
    y_all = y[:, 0]
    snr_all = y[:, 1]
    y_l = y_all[idx_l].astype(int)
    y_u = y_all[idx_u].astype(int)
    snr_u = snr_all[idx_u].astype(int)
    return y_l, y_u, snr_u


def build_prototypes(f_l: np.ndarray, y_l: np.ndarray):
    classes = np.unique(y_l)
    protos = []
    for c in classes:
        feats = f_l[y_l == c]
        p = feats.mean(axis=0, keepdims=False)
        protos.append(p)
    protos = np.stack(protos, axis=0).astype(np.float32)
    protos = l2norm(protos, axis=1)
    return classes, protos


def similarity_to_protos(f_u: np.ndarray, protos: np.ndarray):
    return (f_u @ protos.T).astype(np.float32)


def top1_top2_margin(s: np.ndarray):
    part = np.argpartition(-s, kth=1, axis=1)[:, :2]
    v2 = np.take_along_axis(s, part, axis=1)
    order = np.argsort(-v2, axis=1)
    top_idx = np.take_along_axis(part, order, axis=1)
    top_val = np.take_along_axis(v2, order, axis=1)
    top1_idx = top_idx[:, 0]
    top1_val = top_val[:, 0]
    top2_val = top_val[:, 1]
    margin = top1_val - top2_val
    return top1_idx, top1_val, top2_val, margin


def acc_by_snr(y_true, y_pred, snr):
    out = {}
    for s in sorted(np.unique(snr).tolist()):
        m = (snr == s)
        if m.sum() == 0:
            continue
        out[int(s)] = float(accuracy_score(y_true[m], y_pred[m]))
    return out


# ======================
# 主流程
# ======================
def main(dataset_name="radio128", img_dir=None, iq_dir=None, out_dir=None):
    cfg = get_cfg(dataset_name)
    y_path = cfg["y_path"]
    img_dir = img_dir or cfg["data_dir"]
    iq_dir = iq_dir or cfg["iq_dir"]
    out_dir = out_dir or cfg["fusion_dir"]

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading embeddings for {dataset_name} ...")
    idx_l, idx_u, f_l_img, f_u_img, f_l_iq, f_u_iq = load_embeddings(img_dir, iq_dir)
    y_l, y_u, snr_u = load_labels(y_path, idx_l, idx_u)

    classes_img, protos_img = build_prototypes(f_l_img, y_l)
    classes_iq, protos_iq = build_prototypes(f_l_iq, y_l)

    if not np.array_equal(classes_img, classes_iq):
        raise ValueError("class set mismatch between domains. 请检查两域 labeled 的标签集合是否一致。")
    classes = classes_img
    c_num = len(classes)
    print(f"Classes: {c_num}")

    s_img = similarity_to_protos(f_u_img, protos_img)
    s_iq = similarity_to_protos(f_u_iq, protos_iq)

    top1_idx_img, top1_val_img, top2_val_img, m_img = top1_top2_margin(s_img)
    top1_idx_iq, top1_val_iq, top2_val_iq, m_iq = top1_top2_margin(s_iq)

    w = softmax2(np.stack([m_iq, m_img], axis=1), T=TEMPERATURE)
    w_iq = w[:, 0:1]
    w_img = w[:, 1:2]

    s_f = w_iq * s_iq + w_img * s_img
    pred_idx = np.argmax(s_f, axis=1)
    y_pred = classes[pred_idx].astype(int)

    if USE_REJECT:
        _, top1_val_f, top2_val_f, m_f = top1_top2_margin(s_f)
        reject_mask = (top1_val_f < REJECT_MAXSIM_TH) | (m_f < REJECT_MARGIN_TH)
        y_pred = y_pred.copy()
        y_pred[reject_mask] = REJECT_LABEL
        print(f"Reject enabled. reject_rate={reject_mask.mean():.4f}")

    acc_f = float(accuracy_score(y_u, y_pred))
    acc_f_snr = acc_by_snr(y_u, y_pred, snr_u)

    y_pred_img = classes[top1_idx_img].astype(int)
    y_pred_iq = classes[top1_idx_iq].astype(int)
    acc_img = float(accuracy_score(y_u, y_pred_img))
    acc_iq = float(accuracy_score(y_u, y_pred_iq))
    acc_img_snr = acc_by_snr(y_u, y_pred_img, snr_u)
    acc_iq_snr = acc_by_snr(y_u, y_pred_iq, snr_u)

    print("========== RESULTS ==========")
    print(f"IMG-only acc  : {acc_img:.6f}")
    print(f"IQ-only  acc  : {acc_iq:.6f}")
    print(f"FUSION  acc   : {acc_f:.6f}")
    print("Acc by SNR (fusion):", acc_f_snr)

    np.save(os.path.join(out_dir, "pred_fusion.npy"), y_pred.astype(np.int16))
    np.save(os.path.join(out_dir, "y_true.npy"), y_u.astype(np.int16))
    np.save(os.path.join(out_dir, "snr_u.npy"), snr_u.astype(np.int16))
    np.save(os.path.join(out_dir, "weights_iq_img.npy"), np.concatenate([w_iq, w_img], axis=1).astype(np.float32))

    result = {
        "dataset_name": dataset_name,
        "y_path": y_path,
        "temperature": float(TEMPERATURE),
        "use_reject": bool(USE_REJECT),
        "reject_maxsim_th": float(REJECT_MAXSIM_TH),
        "reject_margin_th": float(REJECT_MARGIN_TH),
        "reject_label": int(REJECT_LABEL),
        "acc_img_only": acc_img,
        "acc_iq_only": acc_iq,
        "acc_fusion": acc_f,
        "acc_by_snr_img_only": acc_img_snr,
        "acc_by_snr_iq_only": acc_iq_snr,
        "acc_by_snr_fusion": acc_f_snr,
        "num_unlabeled": int(len(y_u)),
        "num_classes": int(c_num),
        "img_dir": img_dir,
        "iq_dir": iq_dir,
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "fusion_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved to:", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alldb_radio128",
                        choices=list(DATASET_CONFIG_KEYS))
    parser.add_argument("--img_dir", type=str, default=None,
                        help="图像特征目录（读 F_*.npy / idx_*.npy），默认 data_dir")
    parser.add_argument("--iq_dir", type=str, default=None, help="IQ 特征目录，默认 iq_dir")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录，默认 fusion_dir")
    args = parser.parse_args()
    main(args.dataset, img_dir=args.img_dir, iq_dir=args.iq_dir, out_dir=args.out_dir)
