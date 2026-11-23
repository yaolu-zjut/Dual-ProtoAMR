#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DB_ce.py —— 双域对比学习 + 类别分类头(CE 辅助监督) + 原型融合

改动（相对原版，向后兼容）：新增命令行 --dataset / --seed / --device / --epochs / --lambda_cls，
不传时行为与原来完全一致（alldb_radio128, seed=42, epochs=50, lambda_cls=0.5）；
并把结果写入 out_dir/fusion_result_<dataset>_seed<seed>.json。
"""

import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import get_cfg


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ======================
# 超参
# ======================
SEED = 42
BATCH_SIZE = 128
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.5
ALPHA = 0.5  # 推理期融合权重（IQ 权重）
TRAIN_ALPHA = 0.5  # 训练期融合权重（IQ 权重）：1.0=CE 只走信号支路，0.0=只走图像支路


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ======================
# 分块投影（省显存）
# ======================
PROJ_CHUNK = 32768  # radio1024 的整块图像特征 675840x2048 ≈ 5.5GB，整块上卡必 OOM

def project_in_chunks(head, arr, chunk=PROJ_CHUNK):
    """等价于 head(torch.from_numpy(arr).to(DEVICE))，但按行分块搬到 GPU。

    投影头是逐样本 MLP、归一化也是逐样本，分块不改变结果，只是把峰值显存
    从「整块特征」降到「chunk 行特征」（约 268MB）。返回值留在 CPU 上。
    """
    head.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(arr), chunk):
            x = torch.from_numpy(np.ascontiguousarray(arr[i:i + chunk])).to(DEVICE, dtype=torch.float32)
            outs.append(head(x).detach().cpu())
    return torch.cat(outs, dim=0)

# ======================
# 数据集
# ======================
class IQImgContrastDataset(Dataset):
    def __init__(self, f_iq, f_img, y):
        self.f_iq = f_iq.astype(np.float32)
        self.f_img = f_img.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.f_iq)

    def __getitem__(self, idx):
        return self.f_iq[idx], self.f_img[idx], self.y[idx]


# ======================
# 投影 head
# ======================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, dim=-1)
class FusionClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=11):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, iq_proj, img_proj):
        # 两个模态在公共空间里融合（训练期权重可调）
        z = TRAIN_ALPHA * iq_proj + (1.0 - TRAIN_ALPHA) * img_proj
        z = F.normalize(z, dim=-1)
        logits = self.fc(z)
        return logits


# ======================
# 对比学习损失
# ======================
def contrastive_loss(iq_proj, img_proj, temperature=TEMPERATURE):
    batch_size = iq_proj.shape[0]
    sim_matrix = torch.matmul(iq_proj, img_proj.T) / temperature
    labels = torch.arange(batch_size).to(iq_proj.device)
    loss_iq2img = F.cross_entropy(sim_matrix, labels)
    loss_img2iq = F.cross_entropy(sim_matrix.T, labels)
    return (loss_iq2img + loss_img2iq) / 2

# ======================
# 对比学习训练
# ======================
def train_contrastive_supervised(
    f_l_iq,
    f_l_img,
    y_l,
    out_dir,
    epochs=50,
    lambda_cls=0.5,
    lr=1e-3
):
    os.makedirs(out_dir, exist_ok=True)

    # y_l 可能是 (N, 2)，第一列是类别，第二列是 SNR
    y_l_classes = y_l[:, 0] if y_l.ndim > 1 else y_l
    y_l_classes = y_l_classes.astype(np.int64)

    classes = np.unique(y_l_classes)
    num_classes = int(len(classes))

    # 如果类别标签本身就是 0,1,...,C-1，可以直接用
    # 如果不是连续标签，做一次映射，避免 CrossEntropyLoss 报错
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_l_mapped = np.array([class_to_idx[c] for c in y_l_classes], dtype=np.int64)

    dataset = IQImgContrastDataset(f_l_iq, f_l_img, y_l_mapped)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    iq_head = ProjectionHead(input_dim=f_l_iq.shape[1]).to(DEVICE)
    img_head = ProjectionHead(input_dim=f_l_img.shape[1]).to(DEVICE)
    classifier = FusionClassifier(input_dim=64, num_classes=num_classes).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(iq_head.parameters()) +
        list(img_head.parameters()) +
        list(classifier.parameters()),
        lr=lr
    )

    ce_loss = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        iq_head.train()
        img_head.train()
        classifier.train()

        total_loss = 0.0
        total_con = 0.0
        total_cls = 0.0
        total = 0
        correct = 0

        for iq_batch, img_batch, yb in loader:
            iq_batch = iq_batch.to(DEVICE, dtype=torch.float32)
            img_batch = img_batch.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.long)

            iq_proj = iq_head(iq_batch)
            img_proj = img_head(img_batch)

            # 1) 跨模态对比损失
            loss_con = contrastive_loss(iq_proj, img_proj)

            # 2) 类别监督分类损失
            logits = classifier(iq_proj, img_proj)
            loss_cls = ce_loss(logits, yb)

            # 3) 总损失
            loss = loss_con + lambda_cls * loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = iq_batch.size(0)
            total_loss += loss.item() * bs
            total_con += loss_con.item() * bs
            total_cls += loss_cls.item() * bs
            total += bs

            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()

        print(
            f"[Epoch {epoch}] "
            f"loss={total_loss / total:.4f}, "
            f"con={total_con / total:.4f}, "
            f"cls={total_cls / total:.4f}, "
            f"train_acc={correct / total:.4f}"
        )

    torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head_sup.pth"))
    torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head_sup.pth"))
    torch.save(classifier.state_dict(), os.path.join(out_dir, "fusion_classifier_sup.pth"))

    print("Supervised projection heads saved.")
    return iq_head, img_head, classifier




# ======================
# 混淆矩阵绘制
# ======================
def plot_confusion_matrix(y_true, y_pred, classes, save_path):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=classes
    )

    # 转百分比（按真实类别归一化）
    cm_percent = (
        cm.astype(float)
        /
        (cm.sum(axis=1, keepdims=True)+1e-12)
        * 100
    )

    class_names = [
        "WBFM",
        "QPSK",
        "QAM64",
        "QAM16",
        "PAM4",
        "GFSK",
        "CPFSK",
        "BPSK",
        "8PSK",
        "AM-SSB",
        "AM-DSB"
    ]

    plt.figure(figsize=(12,11))

    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        square=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14}
    )

    plt.xticks(
        fontsize=15,
        rotation=45,
        ha="right",
        fontweight="bold"
    )

    plt.yticks(
        fontsize=15,
        rotation=0,
        fontweight="bold"
    )
    # plt.xlabel(
    #     "Predicted label",
    #     fontsize=14
    # )
    #
    # plt.ylabel(
    #     "True label",
    #     fontsize=14
    # )


    plt.tight_layout()


    # 保存PDF
    plt.savefig(
        save_path,
        bbox_inches="tight",
        format="pdf"
    )

    plt.close()


    print("Saved confusion matrix:",
          save_path)

# ======================
# 融合分类
# ======================
def fusion_classification(f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head):
    iq_head.eval()
    img_head.eval()
    with torch.no_grad():
        f_l_iq_proj = project_in_chunks(iq_head, f_l_iq)
        f_l_img_proj = project_in_chunks(img_head, f_l_img)
        f_u_iq_proj = project_in_chunks(iq_head, f_u_iq)
        f_u_img_proj = project_in_chunks(img_head, f_u_img)

    # 只取标签第一列
    y_l_classes = y_l[:,0] if y_l.ndim>1 else y_l
    y_u_classes = y_u[:,0] if y_u.ndim>1 else y_u

    classes = np.unique(y_l_classes)
    protos_iq = torch.stack([f_l_iq_proj[y_l_classes==c].mean(dim=0) for c in classes])
    protos_img = torch.stack([f_l_img_proj[y_l_classes==c].mean(dim=0) for c in classes])

    s_iq = F.normalize(f_u_iq_proj) @ F.normalize(protos_iq).T
    s_img = F.normalize(f_u_img_proj) @ F.normalize(protos_img).T

    s_f = ALPHA * s_iq + (1-ALPHA) * s_img
    pred_idx = s_f.argmax(dim=1)
    y_pred = classes[pred_idx.cpu().numpy()]

    acc = (y_pred == y_u_classes).mean()
    print("Fusion Accuracy:", acc)

    # ==========================
    # 绘制混淆矩阵
    # ==========================

    classes = np.unique(
        np.concatenate(
            [y_u_classes, y_pred]
        )
    )

    plot_confusion_matrix(
        y_u_classes,
        y_pred,
        classes,
        "confusion_matrix_fusion.pdf"
    )

    return acc, y_pred


# ======================
# 主流程
# ======================
def main(dataset_name="alldb_radio128", seed=SEED, device=None, epochs=50,
         lambda_cls=0.5, out_tag=None, train_alpha=0.5, temp=None):
    global DEVICE, TRAIN_ALPHA, TEMPERATURE
    if device is not None:
        DEVICE = device
    TRAIN_ALPHA = float(train_alpha)
    if temp is not None:
        TEMPERATURE = float(temp)
    set_seed(seed)

    cfg = get_cfg(dataset_name)

    data_dir = cfg["data_dir"]
    iq_dir = cfg["iq_dir"]
    base_dir = cfg["base_dir"]

    tag = out_tag if out_tag else f"{dataset_name}_seed{seed}"
    out_dir = os.path.join(base_dir, "contrastive_heads_ce", "tune_%s" % tag)
    os.makedirs(out_dir, exist_ok=True)

    # 图像 embedding
    f_l_img = np.load(os.path.join(data_dir, "F_labeled.npy"))
    f_u_img = np.load(os.path.join(data_dir, "F_unlabeled.npy"))

    # IQ embedding
    f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy"))
    f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy"))

    # 标签
    y_l = np.load(os.path.join(data_dir, "Y_labeled_20.npy"))
    y_u = np.load(os.path.join(data_dir, "Y_unlabeled_80.npy"))

    print(f"[DB_ce] dataset={dataset_name} seed={seed} device={DEVICE} "
          f"img_dim={f_l_img.shape[1]} iq_dim={f_l_iq.shape[1]} "
          f"epochs={epochs} lambda_cls={lambda_cls} alpha={ALPHA}")

    # 对比学习 + 类别分类头训练
    iq_head, img_head, classifier = train_contrastive_supervised(
        f_l_iq,
        f_l_img,
        y_l,
        out_dir,
        epochs=epochs,
        lambda_cls=lambda_cls,
        lr=1e-3,
    )

    # 融合分类
    acc, y_pred = fusion_classification(f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head)

    result = {
        "script": "DB_ce_tune.py",
        "dataset_name": dataset_name,
        "seed": int(seed),
        "epochs": int(epochs),
        "lambda_cls": float(lambda_cls),
        "alpha": float(ALPHA),
        "train_alpha": float(TRAIN_ALPHA),
        "batch_size": int(BATCH_SIZE),
        "temperature": float(TEMPERATURE),
        "img_dim": int(f_l_img.shape[1]),
        "iq_dim": int(f_l_iq.shape[1]),
        "num_unlabeled": int(len(y_u)),
        "overall_acc": float(acc),
        "run_tag": tag,
    }
    save_path = os.path.join(out_dir, "fusion_result_%s.json" % tag)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("RESULT_JSON:", save_path)
    print(f"FINAL_ACC={float(acc):.10f}")
    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alldb_radio128")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--out_tag", type=str, default=None)
    parser.add_argument("--train_alpha", type=float, default=0.5,
                        help="训练期融合权重（IQ 权重）：1.0 = CE 只走信号支路")
    parser.add_argument("--temp", type=float, default=None, help="对比损失温度（默认 0.5）")
    args = parser.parse_args()
    main(args.dataset, seed=args.seed, device=args.device, epochs=args.epochs,
         lambda_cls=args.lambda_cls, out_tag=args.out_tag,
         train_alpha=args.train_alpha, temp=args.temp)
