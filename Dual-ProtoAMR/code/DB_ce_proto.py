#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DB_ce_proto.py  ——  对比损失 + 原型损失（Prototypical Loss）

与 DB_ce.py 的唯一区别：监督项。
  DB_ce.py      : loss = loss_con + lambda_cls * CE( FusionClassifier(z), y )
                  监督信号来自一个临时的线性分类头 Linear(64 -> C)
  DB_ce_proto.py: loss = loss_con + lambda_cls * ProtoLoss(z, y)
                  监督信号直接来自「到各类质心的距离」，不引入任何额外参数

其中：
  z      = normalize( train_alpha * z_iq + (1 - train_alpha) * z_img )     # 特征级融合
  c_k    = normalize( mean_{i: y_i = k} z_i )                              # batch 内类质心
  logits = cosine(z, c_k) / tau                                            # 到质心的余弦相似度
  ProtoLoss = CrossEntropy(logits, y)

推理端与 DB_ce.py 完全一致（原型分数级融合 s_f = ALPHA*s_iq + (1-ALPHA)*s_img），
因此本文件不改变论文公式 19，只是把训练端的「临时分类器」换成原型损失。

参考：Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.

⚠️ 与 few-shot 的差别（写论文时要说清）：
  1) 本文用「固定类别集 + 全量支持集」而非 episode 采样，每个 batch 的质心是全类别的；
  2) 质心是 batch 内在线估计（online / batch-wise prototype），不是每轮重算的全局原型；
  3) 双模态：质心在融合空间上算，对比损失维持两个模态的可对齐性。

用法（与原脚本完全一致，仅脚本名不同）：
  python DB_ce_proto.py --dataset radio128 --seed 42 --lambda_cls 1.0 \
      --out_dir .../contrastive_heads_ce_proto --tag lambda1.0
"""

import copy
import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from config import get_cfg, DATASET_CONFIG

DATASET_CONFIG_KEYS = sorted(DATASET_CONFIG.keys())


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
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.5
ALPHA = 0.5          # 推理期融合权重：IQ 主导
TRAIN_ALPHA = 0.5    # 训练期（算质心用）特征级融合权重
PROTO_TAU = 0.1      # 原型 logits 的温度；0.1 与 τ=0.07（DB_ce_v2）同量级


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 混淆矩阵输出位置（在 __main__ 里按视图覆盖）
PLOT_DIR = "."
PLOT_TAG = "fusion"


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


# ======================
# 【新增】原型损失（替代 FusionClassifier + CE）
# ======================
def fused_embedding(iq_proj, img_proj, alpha=TRAIN_ALPHA):
    """特征级融合：与 FusionClassifier 里 0.5*iq+0.5*img 同一口径。"""
    z = alpha * iq_proj + (1.0 - alpha) * img_proj
    return F.normalize(z, dim=-1)


def compute_prototypes(z, yb, num_classes):
    """
    batch 内在线求类质心。
    返回 protos (C, D) 与 present (C,) 布尔掩码（该 batch 是否出现该类）。
    未出现的类质心置 0，并由 present 掩码在 loss 中屏蔽。
    """
    d = z.shape[1]
    protos = torch.zeros(num_classes, d, device=z.device, dtype=z.dtype)
    present = torch.zeros(num_classes, device=z.device, dtype=torch.bool)

    # index_add 一次性求和，避免逐类循环
    onehot = F.one_hot(yb, num_classes=num_classes).to(z.dtype)   # (B, C)
    counts = onehot.sum(dim=0)                                    # (C,)
    sums = onehot.T @ z                                           # (C, D)
    present = counts > 0
    safe_counts = counts.clamp(min=1.0).unsqueeze(1)
    protos = sums / safe_counts
    protos = F.normalize(protos, dim=-1)
    protos[~present] = 0.0
    return protos, present


def prototypical_loss(iq_proj, img_proj, yb, num_classes,
                      alpha=TRAIN_ALPHA, tau=PROTO_TAU, return_logits=False):
    """
    原型损失：分类概率 = softmax(cos(z, c_k) / tau)，按真实类取 NLL。
    batch 中未出现的类在 softmax 前被屏蔽（logit 置 -inf）。
    """
    z = fused_embedding(iq_proj, img_proj, alpha)
    protos, present = compute_prototypes(z, yb, num_classes)

    # 余弦相似度；不能直接 normalize 后再乘（原型可能为 0 向量）
    z_n = F.normalize(z, dim=-1)
    logits = z_n @ protos.T / tau                                 # (B, C)

    # 屏蔽本 batch 缺失的类
    logits = logits.masked_fill((~present).unsqueeze(0), float("-inf"))

    loss = F.cross_entropy(logits, yb)
    if return_logits:
        return loss, logits
    return loss


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
# 对比 + 原型 训练
# ======================
def train_contrastive_proto(
    f_l_iq,
    f_l_img,
    y_l,
    out_dir,
    epochs=50,
    lambda_cls=1.0,
    lr=1e-3,
    train_alpha=TRAIN_ALPHA,
    tau=PROTO_TAU,
):
    os.makedirs(out_dir, exist_ok=True)

    # y_l 可能是 (N, 2)，第一列是类别，第二列是 SNR
    y_l_classes = y_l[:, 0] if y_l.ndim > 1 else y_l
    y_l_classes = y_l_classes.astype(np.int64)

    classes = np.unique(y_l_classes)
    num_classes = int(len(classes))

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_l_mapped = np.array([class_to_idx[c] for c in y_l_classes], dtype=np.int64)

    # drop_last=True：保证每个 batch 都覆盖全部类别，质心不会缺行
    # （即便有缺类，prototypical_loss 内部仍会 mask，双保险）
    loader = DataLoader(
        IQImgContrastDataset(f_l_iq, f_l_img, y_l_mapped),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    iq_head = ProjectionHead(input_dim=f_l_iq.shape[1]).to(DEVICE)
    img_head = ProjectionHead(input_dim=f_l_img.shape[1]).to(DEVICE)

    # 注意：没有 classifier —— 原型损失不引入任何额外参数
    optimizer = torch.optim.Adam(
        list(iq_head.parameters()) + list(img_head.parameters()),
        lr=lr
    )

    for epoch in range(1, epochs + 1):
        iq_head.train()
        img_head.train()

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

            # 1) 跨模态对比损失（不变）
            loss_con = contrastive_loss(iq_proj, img_proj)

            # 2) 原型损失（替代原来的 CE + FusionClassifier）
            loss_cls, logits = prototypical_loss(
                iq_proj, img_proj, yb, num_classes,
                alpha=train_alpha, tau=tau, return_logits=True,
            )

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
            f"proto={total_cls / total:.4f}, "
            f"train_acc={correct / total:.4f}"
        )

    torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head_sup.pth"))
    torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head_sup.pth"))
    # 不保存 classifier：原型损失没有分类头

    print("Supervised projection heads saved (prototypical loss, no classifier).")
    return iq_head, img_head


# ======================
# 混淆矩阵绘制
# ======================
def plot_confusion_matrix(y_true, y_pred, classes, save_path):

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # 转百分比（按真实类别归一化）
    cm_percent = (
        cm.astype(float)
        /
        (cm.sum(axis=1, keepdims=True) + 1e-12)
        * 100
    )

    _rml11 = [
        "WBFM", "QPSK", "QAM64", "QAM16", "PAM4",
        "GFSK", "CPFSK", "BPSK", "8PSK", "AM-SSB", "AM-DSB",
    ]
    class_names = _rml11 if len(classes) == len(_rml11) else [str(int(c)) for c in classes]

    plt.figure(figsize=(12, 11))

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

    plt.xticks(fontsize=15, rotation=45, ha="right", fontweight="bold")
    plt.yticks(fontsize=15, rotation=0, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", format="pdf")
    plt.close()

    print("Saved confusion matrix:", save_path)


def _forward_chunked(head, feat, device, chunk=32768):
    """分块前向，避免 1024high 上一次性把 675840×2048 送进显存导致 OOM。
    head 是逐样本的 MLP + normalize，分块结果与整块前向逐位一致。"""
    outs = []
    n = feat.shape[0]
    with torch.no_grad():
        for i in range(0, n, chunk):
            xb = torch.from_numpy(feat[i:i + chunk]).to(device, dtype=torch.float32)
            outs.append(head(xb).cpu())
    return torch.cat(outs, dim=0)


# ======================
# 融合分类（与 DB_ce.py 完全一致：原型分数级融合）
# ======================
def fusion_classification(f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head):
    iq_head.eval()
    img_head.eval()
    # 推理全程在 CPU 上做（特征只有 64 维，规模很小）；
    # 前向分块，避免 1024high 上一次性搬 675840×2048 进显存导致 OOM。
    f_l_iq_proj = _forward_chunked(iq_head, f_l_iq, DEVICE)
    f_l_img_proj = _forward_chunked(img_head, f_l_img, DEVICE)
    f_u_iq_proj = _forward_chunked(iq_head, f_u_iq, DEVICE)
    f_u_img_proj = _forward_chunked(img_head, f_u_img, DEVICE)

    y_l_classes = y_l[:, 0] if y_l.ndim > 1 else y_l
    y_u_classes = y_u[:, 0] if y_u.ndim > 1 else y_u

    classes = np.unique(y_l_classes)
    protos_iq = torch.stack([f_l_iq_proj[y_l_classes == c].mean(dim=0) for c in classes])
    protos_img = torch.stack([f_l_img_proj[y_l_classes == c].mean(dim=0) for c in classes])
    protos_iq = F.normalize(protos_iq)
    protos_img = F.normalize(protos_img)

    # 分数逐块累加，显存/内存峰值只跟 chunk 有关
    n_u = f_u_iq_proj.shape[0]
    pred_idx = torch.empty(n_u, dtype=torch.long)
    for i in range(0, n_u, 65536):
        s_iq = F.normalize(f_u_iq_proj[i:i + 65536]) @ protos_iq.T
        s_img = F.normalize(f_u_img_proj[i:i + 65536]) @ protos_img.T
        s_f = ALPHA * s_iq + (1 - ALPHA) * s_img
        pred_idx[i:i + 65536] = s_f.argmax(dim=1)

    y_pred = classes[pred_idx.numpy()]

    acc = (y_pred == y_u_classes).mean()
    print("Fusion Accuracy:", acc)

    classes_plot = np.unique(np.concatenate([y_u_classes, y_pred]))

    try:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_confusion_matrix(
            y_u_classes,
            y_pred,
            classes_plot,
            os.path.join(PLOT_DIR, f"confusion_matrix_fusion_{PLOT_TAG}.pdf"),
        )
    except Exception as _e:
        print(f"[warn] plot_confusion_matrix skipped: {_e}")

    return acc, y_pred


# ======================
# 【可选】训练侧留出验证：用全局原型给标注集打分（诊断用，不参与调参）
# ======================
def labeled_proto_acc(f_l_iq, f_l_img, y_l, iq_head, img_head, alpha=ALPHA):
    """给标注集本身按全局原型打分，仅用于快速自检/看训练是否收敛。
    分块前向，避免大数据集上显存峰值。"""
    iq_head.eval()
    img_head.eval()
    z_iq = _forward_chunked(iq_head, f_l_iq, "cpu")
    z_img = _forward_chunked(img_head, f_l_img, "cpu")
    yc = y_l[:, 0] if y_l.ndim > 1 else y_l
    classes = np.unique(yc)
    pi = torch.stack([z_iq[yc == c].mean(dim=0) for c in classes])
    pj = torch.stack([z_img[yc == c].mean(dim=0) for c in classes])
    s = alpha * (F.normalize(z_iq) @ F.normalize(pi).T) + \
        (1 - alpha) * (F.normalize(z_img) @ F.normalize(pj).T)
    pred = classes[s.argmax(dim=1).cpu().numpy()]
    return float((pred == yc).mean())


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    _DEV0, _ALPHA0 = DEVICE, ALPHA   # 保存文件级默认值，避免被 args 覆盖后丢失
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="radio128",
                        choices=list(DATASET_CONFIG_KEYS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=_DEV0)
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--iq_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=_ALPHA0)
    parser.add_argument("--train_alpha", type=float, default=TRAIN_ALPHA)
    parser.add_argument("--tau", type=float, default=PROTO_TAU)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    # 顶层定义的模块级全局量（DEVICE/ALPHA/SEED 在 main 里会重新赋值）；
    # 这里不用 global，改成显式回写模块属性，避免 "assigned before global" 语法错误。
    SEED = args.seed
    DEVICE = args.device or _DEV0
    if not torch.cuda.is_available():
        DEVICE = "cpu"
    ALPHA = args.alpha
    globals()["SEED"] = SEED
    globals()["DEVICE"] = DEVICE
    globals()["ALPHA"] = ALPHA
    set_seed(SEED)

    dataset_name = args.dataset
    cfg = get_cfg(dataset_name)

    data_dir = args.data_dir if args.data_dir else cfg["data_dir"]
    iq_dir = args.iq_dir if args.iq_dir else cfg["iq_dir"]
    base_dir = cfg["base_dir"]

    img_dir = args.img_dir if args.img_dir else data_dir
    out_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "contrastive_heads_ce_proto")
    os.makedirs(out_dir, exist_ok=True)

    PLOT_DIR = out_dir
    PLOT_TAG = args.tag or f"seed{SEED}"

    print(f"[DB_ce_proto.py] dataset={dataset_name} seed={SEED} device={DEVICE} "
          f"lambda_cls={args.lambda_cls} train_alpha={args.train_alpha} tau={args.tau}")
    print(f"  img_dir={img_dir}")
    print(f"  data_dir={data_dir}")
    print(f"  iq_dir={iq_dir}")

    f_l_img = np.load(os.path.join(img_dir, "F_labeled.npy"))
    f_u_img = np.load(os.path.join(img_dir, "F_unlabeled.npy"))

    f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy"))
    f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy"))

    y_l = np.load(os.path.join(data_dir, "Y_labeled_20.npy"))
    y_u = np.load(os.path.join(data_dir, "Y_unlabeled_80.npy"))

    # 对比损失 + 原型损失 训练
    iq_head, img_head = train_contrastive_proto(
        f_l_iq,
        f_l_img,
        y_l,
        out_dir,
        epochs=args.epochs,
        lambda_cls=args.lambda_cls,
        lr=args.lr,
        train_alpha=args.train_alpha,
        tau=args.tau,
    )

    # 诊断：标注集上的全局原型自洽精度（训练集精度，仅供参考）
    # 全局原型打分本身与设备无关；这里显式把 head 搬回 CPU，
    # 避免 DEVICE=cuda:N 时 head 在卡上、特征在 CPU 上导致 device 不匹配。
    iq_head_c = copy.deepcopy(iq_head).to("cpu")
    img_head_c = copy.deepcopy(img_head).to("cpu")
    lacc = labeled_proto_acc(f_l_iq, f_l_img, y_l, iq_head_c, img_head_c)
    print(f"LABELED_PROTO_ACC={lacc:.4f}")

    # 融合分类（无标注集，与 DB_ce.py 同一评测口径）
    acc, y_pred = fusion_classification(
        f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head
    )

    np.save(os.path.join(out_dir, "pred_proto_fusion.npy"), y_pred.astype(np.int16))

    result = {
        "dataset_name": dataset_name,
        "method": "DB_ce_proto.py: prototypical loss + cross-modal contrastive loss + prototype fusion",
        "supervised_loss": "prototypical (cosine to batch centroids, tau=%.4g)" % args.tau,
        "seed": int(SEED),
        "img_dir": img_dir,
        "data_dir": data_dir,
        "iq_dir": iq_dir,
        "out_dir": out_dir,
        "img_dim": int(f_l_img.shape[1]),
        "iq_dim": int(f_l_iq.shape[1]),
        "epochs": int(args.epochs),
        "lambda_proto": float(args.lambda_cls),
        "lr": float(args.lr),
        "alpha": float(ALPHA),
        "train_alpha": float(args.train_alpha),
        "proto_tau": float(args.tau),
        "labeled_proto_acc": float(lacc),
        "overall_acc": float(acc),
        "tag": args.tag,
    }
    name = (f"fusion_result_proto_{dataset_name}_seed{SEED}"
            + (f"_{args.tag}" if args.tag else "") + ".json")
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"FINAL_ACC={float(acc)}")
    print("Saved to:", os.path.join(out_dir, name))
