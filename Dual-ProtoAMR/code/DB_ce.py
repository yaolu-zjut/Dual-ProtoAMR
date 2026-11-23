#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
ALPHA = 0.5  # 融合权重：IQ主导


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
class FusionClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=11):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, iq_proj, img_proj):
        # 两个模态在公共空间里融合
        z = 0.5 * iq_proj + 0.5 * img_proj
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

    # 类别名动态生成：优先用 RadioML 2016.10a 的 11 类名，
    # 类数不匹配时（如 alldb 的 12 类）回退成类索引，避免 seaborn 因长度不等报错
    _rml11 = [
        "WBFM", "QPSK", "QAM64", "QAM16", "PAM4",
        "GFSK", "CPFSK", "BPSK", "8PSK", "AM-SSB", "AM-DSB",
    ]
    class_names = _rml11 if len(classes) == len(_rml11) else [str(int(c)) for c in classes]

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
        f_l_iq_proj = iq_head(torch.from_numpy(f_l_iq).to(DEVICE, dtype=torch.float32))
        f_l_img_proj = img_head(torch.from_numpy(f_l_img).to(DEVICE, dtype=torch.float32))
        f_u_iq_proj = iq_head(torch.from_numpy(f_u_iq).to(DEVICE, dtype=torch.float32))
        f_u_img_proj = img_head(torch.from_numpy(f_u_img).to(DEVICE, dtype=torch.float32))

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

    # 绘图出错（例如类数与内置类名不一致）不应中断整个实验
    try:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_confusion_matrix(
            y_u_classes,
            y_pred,
            classes,
            os.path.join(PLOT_DIR, f"confusion_matrix_fusion_{PLOT_TAG}.pdf"),
        )
    except Exception as _e:
        print(f"[warn] plot_confusion_matrix skipped: {_e}")

    return acc, y_pred



def classifier_evaluation(f_l_iq, f_u_iq, f_l_img, f_u_img, y_l, y_u,
                          iq_head, img_head, classifier, name="Classifier"):
    """用训好的 FusionClassifier 直接分类。

    原 DB_ce.py 在评测里把分类头丢弃了（只用了原型余弦），本函数补上这一步。
    类别索引与 train_contrastive_supervised / fusion_classification 一致：
    index i <-> np.unique(y_l 类别列)[i]
    """
    iq_head.eval()
    img_head.eval()
    classifier.eval()

    y_l_classes = y_l[:, 0] if y_l.ndim > 1 else y_l
    y_u_classes = y_u[:, 0] if y_u.ndim > 1 else y_u
    classes = np.unique(y_l_classes)

    with torch.no_grad():
        f_u_iq_proj = iq_head(torch.from_numpy(f_u_iq).to(DEVICE, dtype=torch.float32))
        f_u_img_proj = img_head(torch.from_numpy(f_u_img).to(DEVICE, dtype=torch.float32))
        logits = classifier(f_u_iq_proj, f_u_img_proj)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()

    y_pred = classes[pred_idx]
    acc = (y_pred == y_u_classes).mean()
    print(f"{name} Accuracy:", acc)
    return acc, y_pred


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    # out_dir = "/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/contrastive_heads"
    #
    # # 图像 embedding
    # f_l_img = np.load("/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/data/F_labeled.npy")
    # f_u_img = np.load("/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/data/F_unlabeled.npy")
    #
    # # IQ embedding
    # f_l_iq = np.load("/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/data_iq/F_labeled_iq.npy")
    # f_u_iq = np.load("/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/data_iq/F_unlabeled_iq.npy")
    #
    # # 标签
    # y_l = np.load("/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/data/Y_labeled_20.npy")
    # y_u = np.load("/public/mount_data/data/yaolu/signal_group/ghy/zero_short/radio512_lowsnr/data/Y_unlabeled_80.npy")


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="radio128",
                        choices=list(DATASET_CONFIG_KEYS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None, help="图像特征目录，默认 data_dir")
    parser.add_argument("--iq_dir", type=str, default=None, help="IQ 特征目录，默认 iq_dir")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="输出目录，默认 base_dir/contrastive_heads_ce")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    SEED = args.seed
    if args.device:
        DEVICE = args.device
    if not torch.cuda.is_available():
        DEVICE = "cpu"
    ALPHA = args.alpha
    set_seed(SEED)

    dataset_name = args.dataset
    cfg = get_cfg(dataset_name)

    data_dir = cfg["data_dir"]
    iq_dir = args.iq_dir if args.iq_dir else cfg["iq_dir"]
    base_dir = cfg["base_dir"]

    img_dir = args.img_dir if args.img_dir else data_dir
    out_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "contrastive_heads_ce")
    os.makedirs(out_dir, exist_ok=True)

    PLOT_DIR = out_dir
    PLOT_TAG = args.tag or f"seed{SEED}"

    print(f"[DB_ce.py] dataset={dataset_name} seed={SEED} device={DEVICE} "
          f"img_dir={img_dir} out_dir={out_dir}")

    # 图像 embedding（按视图隔离，来自 img_dir）
    f_l_img = np.load(os.path.join(img_dir, "F_labeled.npy"))
    f_u_img = np.load(os.path.join(img_dir, "F_unlabeled.npy"))

    # IQ embedding
    f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy"))
    f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy"))

    # 标签
    y_l = np.load(os.path.join(data_dir, "Y_labeled_20.npy"))
    y_u = np.load(os.path.join(data_dir, "Y_unlabeled_80.npy"))

    # 对比学习 + 类别监督 训练
    iq_head, img_head, classifier = train_contrastive_supervised(
        f_l_iq,
        f_l_img,
        y_l,
        out_dir,
        epochs=args.epochs,
        lambda_cls=args.lambda_cls,
        lr=args.lr,
    )

    # 融合分类（原型）
    acc, y_pred = fusion_classification(
        f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head
    )

    np.save(os.path.join(out_dir, "pred_proto_fusion.npy"), y_pred.astype(np.int16))

    # 类别分类头（原来被丢弃的那一个）
    acc_clf, y_pred_clf = classifier_evaluation(
        f_l_iq, f_u_iq, f_l_img, f_u_img, y_l, y_u,
        iq_head, img_head, classifier,
        name="[DB_ce] Classifier Head",
    )
    np.save(os.path.join(out_dir, "pred_classifier.npy"), y_pred_clf.astype(np.int16))

    result = {
        "dataset_name": dataset_name,
        "method": "DB_ce.py: cross-entropy loss + cross-modal contrastive loss + prototype fusion",
        "seed": int(SEED),
        "img_dir": img_dir,
        "out_dir": out_dir,
        "img_dim": int(f_l_img.shape[1]),
        "iq_dim": int(f_l_iq.shape[1]),
        "epochs": int(args.epochs),
        "lambda_cls": float(args.lambda_cls),
        "lr": float(args.lr),
        "alpha": float(ALPHA),
        "overall_acc": float(acc),
        "overall_acc_classifier": float(acc_clf),
        "overall_acc_best": float(max(acc, acc_clf)),
        "tag": args.tag,
    }
    name = f"fusion_result_{dataset_name}_seed{SEED}" + (f"_{args.tag}" if args.tag else "") + ".json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"FINAL_ACC={float(acc)}")
    print("Saved to:", os.path.join(out_dir, name))
