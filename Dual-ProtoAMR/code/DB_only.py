# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Prototype-only ablation for dual-domain contrastive framework.
#
# Purpose:
# 1) Keep the paper's inference stage unchanged: class prototypes + IQ/Image similarity fusion.
# 2) Add a CE-only supervised training branch only as an ablation.
# 3) The CE head is used only to provide training loss; it is NOT used for final inference.
#
# Compared branches:
# - Method: Contrastive + CE auxiliary supervision -> Prototype Fusion Inference
# - Baseline: CE-only auxiliary supervision, no contrastive alignment -> Prototype Fusion Inference
# """
#
# import os
# import random
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from config import get_cfg
#
#
# # ======================
# # Hyperparameters
# # ======================
# SEED = 42
# BATCH_SIZE = 128
# DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"
# TEMPERATURE = 0.5
# ALPHA = 0.5  # prototype fusion weight: s = alpha * s_iq + (1-alpha) * s_img
#
#
# def set_seed(seed=SEED):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#
#
# # ======================
# # Dataset
# # ======================
# class IQImgLabelDataset(Dataset):
#     def __init__(self, f_iq, f_img, y):
#         self.f_iq = f_iq.astype(np.float32)
#         self.f_img = f_img.astype(np.float32)
#         self.y = y.astype(np.int64)
#
#     def __len__(self):
#         return len(self.f_iq)
#
#     def __getitem__(self, idx):
#         return self.f_iq[idx], self.f_img[idx], self.y[idx]
#
#
# # ======================
# # Models
# # ======================
# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim),
#         )
#
#     def forward(self, x):
#         x = self.net(x)
#         return F.normalize(x, dim=-1)
#
#
# class TrainOnlyCEHead(nn.Module):
#     """Auxiliary CE head used only during training, not during final inference."""
#
#     def __init__(self, input_dim=64, num_classes=11):
#         super().__init__()
#         self.fc = nn.Linear(input_dim, num_classes)
#
#     def forward(self, iq_proj, img_proj):
#         z = 0.5 * iq_proj + 0.5 * img_proj
#         z = F.normalize(z, dim=-1)
#         return self.fc(z)
#
#
# # ======================
# # Utilities
# # ======================
# def get_class_column(y):
#     return y[:, 0] if y.ndim > 1 else y
#
#
# def build_label_mapping(y_l):
#     y_l_classes = get_class_column(y_l).astype(np.int64)
#     classes = np.unique(y_l_classes)
#     class_to_idx = {c: i for i, c in enumerate(classes)}
#     y_l_mapped = np.array([class_to_idx[c] for c in y_l_classes], dtype=np.int64)
#     return classes, y_l_mapped
#
#
# def contrastive_loss(iq_proj, img_proj, temperature=TEMPERATURE):
#     batch_size = iq_proj.shape[0]
#     sim_matrix = torch.matmul(iq_proj, img_proj.T) / temperature
#     labels = torch.arange(batch_size, device=iq_proj.device)
#     loss_iq2img = F.cross_entropy(sim_matrix, labels)
#     loss_img2iq = F.cross_entropy(sim_matrix.T, labels)
#     return 0.5 * (loss_iq2img + loss_img2iq)
#
#
# @torch.no_grad()
# def prototype_fusion_classification(
#     f_l_iq,
#     f_l_img,
#     y_l,
#     f_u_iq,
#     f_u_img,
#     y_u,
#     iq_head,
#     img_head,
#     alpha=ALPHA,
#     tag="Prototype",
# ):
#     """Paper-consistent inference: prototypes + similarity fusion + argmax."""
#     iq_head.eval()
#     img_head.eval()
#
#     f_l_iq_proj = iq_head(torch.from_numpy(f_l_iq).to(DEVICE, dtype=torch.float32))
#     f_l_img_proj = img_head(torch.from_numpy(f_l_img).to(DEVICE, dtype=torch.float32))
#     f_u_iq_proj = iq_head(torch.from_numpy(f_u_iq).to(DEVICE, dtype=torch.float32))
#     f_u_img_proj = img_head(torch.from_numpy(f_u_img).to(DEVICE, dtype=torch.float32))
#
#     y_l_classes = get_class_column(y_l).astype(np.int64)
#     y_u_classes = get_class_column(y_u).astype(np.int64)
#
#     classes = np.unique(y_l_classes)
#     protos_iq = torch.stack([
#         f_l_iq_proj[y_l_classes == c].mean(dim=0) for c in classes
#     ])
#     protos_img = torch.stack([
#         f_l_img_proj[y_l_classes == c].mean(dim=0) for c in classes
#     ])
#
#     s_iq = F.normalize(f_u_iq_proj, dim=-1) @ F.normalize(protos_iq, dim=-1).T
#     s_img = F.normalize(f_u_img_proj, dim=-1) @ F.normalize(protos_img, dim=-1).T
#
#     s_f = alpha * s_iq + (1.0 - alpha) * s_img
#     pred_idx = s_f.argmax(dim=1)
#     y_pred = classes[pred_idx.cpu().numpy()]
#
#     acc = float((y_pred == y_u_classes).mean())
#     print(f"[{tag}] Prototype Fusion Accuracy: {acc:.10f}")
#     return acc, y_pred
#
#
# # ======================
# # Branch 1: Method, Contrastive + CE auxiliary loss
# # ======================
# def train_contrastive_ce_for_prototype(
#     f_l_iq,
#     f_l_img,
#     y_l,
#     out_dir,
#     epochs=50,
#     lambda_cls=0.5,
#     lr=1e-3,
# ):
#     os.makedirs(out_dir, exist_ok=True)
#     classes, y_l_mapped = build_label_mapping(y_l)
#     num_classes = len(classes)
#
#     dataset = IQImgLabelDataset(f_l_iq, f_l_img, y_l_mapped)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
#
#     iq_head = ProjectionHead(input_dim=f_l_iq.shape[1]).to(DEVICE)
#     img_head = ProjectionHead(input_dim=f_l_img.shape[1]).to(DEVICE)
#     ce_head = TrainOnlyCEHead(input_dim=64, num_classes=num_classes).to(DEVICE)
#
#     optimizer = torch.optim.Adam(
#         list(iq_head.parameters()) + list(img_head.parameters()) + list(ce_head.parameters()),
#         lr=lr,
#     )
#     ce_loss = nn.CrossEntropyLoss()
#
#     for epoch in range(1, epochs + 1):
#         iq_head.train()
#         img_head.train()
#         ce_head.train()
#
#         total_loss = total_con = total_cls = 0.0
#         total = correct = 0
#
#         for iq_batch, img_batch, yb in loader:
#             iq_batch = iq_batch.to(DEVICE, dtype=torch.float32)
#             img_batch = img_batch.to(DEVICE, dtype=torch.float32)
#             yb = yb.to(DEVICE, dtype=torch.long)
#
#             iq_proj = iq_head(iq_batch)
#             img_proj = img_head(img_batch)
#
#             loss_con = contrastive_loss(iq_proj, img_proj)
#             logits = ce_head(iq_proj, img_proj)
#             loss_cls = ce_loss(logits, yb)
#             loss = loss_con + lambda_cls * loss_cls
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             bs = iq_batch.size(0)
#             total_loss += loss.item() * bs
#             total_con += loss_con.item() * bs
#             total_cls += loss_cls.item() * bs
#             total += bs
#             correct += (logits.argmax(dim=1) == yb).sum().item()
#
#         print(
#             f"[Method: Contrastive+CE][Epoch {epoch}] "
#             f"loss={total_loss / total:.4f}, "
#             f"con={total_con / total:.4f}, "
#             f"cls={total_cls / total:.4f}, "
#             f"train_acc={correct / total:.4f}"
#         )
#
#     torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head_con_ce.pth"))
#     torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head_con_ce.pth"))
#     torch.save(ce_head.state_dict(), os.path.join(out_dir, "train_only_ce_head_con_ce.pth"))
#     np.save(os.path.join(out_dir, "classes.npy"), classes)
#     print("Method models saved. Final inference will use prototypes, not the CE head.")
#     return iq_head, img_head
#
#
# # ======================
# # Branch 2: CE-only supervised ablation, no contrastive loss
# # ======================
# def train_ce_only_for_prototype(
#     f_l_iq,
#     f_l_img,
#     y_l,
#     out_dir,
#     epochs=50,
#     lr=1e-3,
# ):
#     os.makedirs(out_dir, exist_ok=True)
#     classes, y_l_mapped = build_label_mapping(y_l)
#     num_classes = len(classes)
#
#     dataset = IQImgLabelDataset(f_l_iq, f_l_img, y_l_mapped)
#     loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
#
#     iq_head = ProjectionHead(input_dim=f_l_iq.shape[1]).to(DEVICE)
#     img_head = ProjectionHead(input_dim=f_l_img.shape[1]).to(DEVICE)
#     ce_head = TrainOnlyCEHead(input_dim=64, num_classes=num_classes).to(DEVICE)
#
#     optimizer = torch.optim.Adam(
#         list(iq_head.parameters()) + list(img_head.parameters()) + list(ce_head.parameters()),
#         lr=lr,
#     )
#     ce_loss = nn.CrossEntropyLoss()
#
#     for epoch in range(1, epochs + 1):
#         iq_head.train()
#         img_head.train()
#         ce_head.train()
#
#         total_loss = 0.0
#         total = correct = 0
#
#         for iq_batch, img_batch, yb in loader:
#             iq_batch = iq_batch.to(DEVICE, dtype=torch.float32)
#             img_batch = img_batch.to(DEVICE, dtype=torch.float32)
#             yb = yb.to(DEVICE, dtype=torch.long)
#
#             iq_proj = iq_head(iq_batch)
#             img_proj = img_head(img_batch)
#
#             # CE-only ablation: no cross-modal contrastive alignment term.
#             logits = ce_head(iq_proj, img_proj)
#             loss = ce_loss(logits, yb)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             bs = iq_batch.size(0)
#             total_loss += loss.item() * bs
#             total += bs
#             correct += (logits.argmax(dim=1) == yb).sum().item()
#
#         print(
#             f"[Ablation: CE-only][Epoch {epoch}] "
#             f"loss={total_loss / total:.4f}, "
#             f"train_acc={correct / total:.4f}"
#         )
#
#     torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head_ce_only.pth"))
#     torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head_ce_only.pth"))
#     torch.save(ce_head.state_dict(), os.path.join(out_dir, "train_only_ce_head_ce_only.pth"))
#     np.save(os.path.join(out_dir, "classes.npy"), classes)
#     print("CE-only ablation models saved. Final inference will use prototypes, not the CE head.")
#     return iq_head, img_head
#
#
# # ======================
# # Main
# # ======================
# if __name__ == "__main__":
#     set_seed(SEED)
#
#     dataset_name = "radio512_lowsnr1"  # change here if needed
#     cfg = get_cfg(dataset_name)
#
#     data_dir = cfg["data_dir"]
#     iq_dir = args.iq_dir if args.iq_dir else cfg["iq_dir"]
#     base_dir = cfg["base_dir"]
#
#     # Image-domain embeddings
#     f_l_img = np.load(os.path.join(data_dir, "F_labeled.npy"))
#     f_u_img = np.load(os.path.join(data_dir, "F_unlabeled.npy"))
#
#     # IQ-domain embeddings
#     f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy"))
#     f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy"))
#
#     # Labels
#     y_l = np.load(os.path.join(data_dir, "Y_labeled_20.npy"))
#     y_u = np.load(os.path.join(data_dir, "Y_unlabeled_80.npy"))
#
#     out_root = os.path.join(base_dir, "prototype_ablation")
#     out_method = os.path.join(out_root, "contrastive_ce")
#     out_ce_only = os.path.join(out_root, "ce_only")
#     os.makedirs(out_root, exist_ok=True)
#
#     print("========== Branch 1: Method, Contrastive + CE auxiliary loss ==========")
#     method_iq_head, method_img_head = train_contrastive_ce_for_prototype(
#         f_l_iq,
#         f_l_img,
#         y_l,
#         out_method,
#         epochs=50,
#         lambda_cls=0.5,
#         lr=1e-3,
#     )
#     acc_method, pred_method = prototype_fusion_classification(
#         f_l_iq,
#         f_l_img,
#         y_l,
#         f_u_iq,
#         f_u_img,
#         y_u,
#         method_iq_head,
#         method_img_head,
#         alpha=ALPHA,
#         tag="Method: Contrastive+CE",
#     )
#     np.save(os.path.join(out_method, "pred_proto_fusion.npy"), pred_method.astype(np.int16))
#
#     print("\n========== Branch 2: CE-only supervised ablation ==========")
#     ce_iq_head, ce_img_head = train_ce_only_for_prototype(
#         f_l_iq,
#         f_l_img,
#         y_l,
#         out_ce_only,
#         epochs=50,
#         lr=1e-3,
#     )
#     acc_ce_only, pred_ce_only = prototype_fusion_classification(
#         f_l_iq,
#         f_l_img,
#         y_l,
#         f_u_iq,
#         f_u_img,
#         y_u,
#         ce_iq_head,
#         ce_img_head,
#         alpha=ALPHA,
#         tag="Ablation: CE-only",
#     )
#     np.save(os.path.join(out_ce_only, "pred_proto_fusion.npy"), pred_ce_only.astype(np.int16))
#
#     print("\n========== Prototype-only comparison ==========")
#     print(f"Method Contrastive+CE -> Prototype Fusion Acc: {acc_method:.10f}")
#     print(f"Ablation CE-only      -> Prototype Fusion Acc: {acc_ce_only:.10f}")
#     print(f"Gain: {acc_method - acc_ce_only:+.10f}")



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


# ======================
# 超参
# ======================
SEED = 42
BATCH_SIZE = 128
DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.5
ALPHA = 0.5  # 融合权重：IQ主导


# ======================
# 随机种子
# ======================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================
# 标签处理
# ======================
def get_class_labels(y):
    """
    y 可能是 (N, 2)，第一列是类别，第二列是 SNR。
    返回类别标签的一维数组。
    """
    y_classes = y[:, 0] if y.ndim > 1 else y
    return y_classes.astype(np.int64)


def build_label_mapping(y_l):
    """
    CrossEntropyLoss 要求标签是 0,1,...,C-1。
    如果原始类别不是连续编号，这里做一次映射。
    """
    y_l_classes = get_class_labels(y_l)
    classes = np.unique(y_l_classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_l_mapped = np.array([class_to_idx[c] for c in y_l_classes], dtype=np.int64)
    return classes, y_l_mapped


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
# 分支1：对比学习 + CE 监督训练
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

    classes, y_l_mapped = build_label_mapping(y_l)
    num_classes = int(len(classes))

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

            # 3) 总损失：对比学习 + CE
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
            f"[Contrastive+CE][Epoch {epoch}] "
            f"loss={total_loss / total:.4f}, "
            f"con={total_con / total:.4f}, "
            f"cls={total_cls / total:.4f}, "
            f"train_acc={correct / total:.4f}"
        )

    torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head_con_ce.pth"))
    torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head_con_ce.pth"))
    torch.save(classifier.state_dict(), os.path.join(out_dir, "fusion_classifier_con_ce.pth"))
    np.save(os.path.join(out_dir, "classes.npy"), classes)

    print("Contrastive+CE models saved.")
    return iq_head, img_head, classifier, classes


# ======================
# 分支2：纯监督 CE 训练，不使用 contrastive_loss
# ======================
def train_supervised_only(
    f_l_iq,
    f_l_img,
    y_l,
    out_dir,
    epochs=50,
    lr=1e-3
):
    os.makedirs(out_dir, exist_ok=True)

    classes, y_l_mapped = build_label_mapping(y_l)
    num_classes = int(len(classes))

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
        total = 0
        correct = 0

        for iq_batch, img_batch, yb in loader:
            iq_batch = iq_batch.to(DEVICE, dtype=torch.float32)
            img_batch = img_batch.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE, dtype=torch.long)

            iq_proj = iq_head(iq_batch)
            img_proj = img_head(img_batch)

            # 纯监督分支：这里只算 CE，不算 contrastive_loss
            logits = classifier(iq_proj, img_proj)
            loss = ce_loss(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = iq_batch.size(0)
            total_loss += loss.item() * bs
            total += bs

            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()

        print(
            f"[CE-only][Epoch {epoch}] "
            f"loss={total_loss / total:.4f}, "
            f"train_acc={correct / total:.4f}"
        )

    torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head_ce_only.pth"))
    torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head_ce_only.pth"))
    torch.save(classifier.state_dict(), os.path.join(out_dir, "fusion_classifier_ce_only.pth"))
    np.save(os.path.join(out_dir, "classes.npy"), classes)

    print("CE-only models saved.")
    return iq_head, img_head, classifier, classes


# ======================
# 评估1：使用训练好的 classifier 直接分类
# ======================
def classifier_evaluation(f_u_iq, f_u_img, y_u, iq_head, img_head, classifier, classes, name="Classifier"):
    iq_head.eval()
    img_head.eval()
    classifier.eval()

    y_u_classes = get_class_labels(y_u)

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
# 评估2：原型分类，保留你原来的评估方式
# ======================
def fusion_classification(f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head, name="Prototype Fusion"):
    iq_head.eval()
    img_head.eval()
    with torch.no_grad():
        f_l_iq_proj = iq_head(torch.from_numpy(f_l_iq).to(DEVICE, dtype=torch.float32))
        f_l_img_proj = img_head(torch.from_numpy(f_l_img).to(DEVICE, dtype=torch.float32))
        f_u_iq_proj = iq_head(torch.from_numpy(f_u_iq).to(DEVICE, dtype=torch.float32))
        f_u_img_proj = img_head(torch.from_numpy(f_u_img).to(DEVICE, dtype=torch.float32))

    # 只取标签第一列
    y_l_classes = get_class_labels(y_l)
    y_u_classes = get_class_labels(y_u)

    classes = np.unique(y_l_classes)
    protos_iq = torch.stack([
        f_l_iq_proj[torch.from_numpy(y_l_classes == c).to(DEVICE)].mean(dim=0)
        for c in classes
    ])
    protos_img = torch.stack([
        f_l_img_proj[torch.from_numpy(y_l_classes == c).to(DEVICE)].mean(dim=0)
        for c in classes
    ])

    s_iq = F.normalize(f_u_iq_proj, dim=-1) @ F.normalize(protos_iq, dim=-1).T
    s_img = F.normalize(f_u_img_proj, dim=-1) @ F.normalize(protos_img, dim=-1).T

    s_f = ALPHA * s_iq + (1 - ALPHA) * s_img
    pred_idx = s_f.argmax(dim=1)
    y_pred = classes[pred_idx.cpu().numpy()]

    acc = (y_pred == y_u_classes).mean()
    print(f"{name} Accuracy:", acc)

    return acc, y_pred


# ======================
# 主流程
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alldb_radio512",
                        choices=list(DATASET_CONFIG_KEYS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--img_dir", type=str, default=None, help="图像特征目录，默认 data_dir")
    parser.add_argument("--iq_dir", type=str, default=None, help="IQ 特征目录，默认 iq_dir")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="输出根目录，默认 base_dir/prototype_ablation")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--branch", type=str, default="both",
                        choices=["both", "ce_only"],
                        help="ce_only 时只跑 CE-only 分支（省一半时间）")
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
    root_out = args.out_dir if args.out_dir else os.path.join(base_dir, "prototype_ablation")
    os.makedirs(root_out, exist_ok=True)

    print(f"[DB_only.py] dataset={dataset_name} seed={SEED} device={DEVICE} "
          f"img_dir={img_dir} out_dir={root_out} branch={args.branch}")

    # 图像 embedding（按视图隔离，来自 img_dir）
    f_l_img = np.load(os.path.join(img_dir, "F_labeled.npy"))
    f_u_img = np.load(os.path.join(img_dir, "F_unlabeled.npy"))

    # IQ embedding
    f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy"))
    f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy"))

    # 标签
    y_l = np.load(os.path.join(data_dir, "Y_labeled_20.npy"))
    y_u = np.load(os.path.join(data_dir, "Y_unlabeled_80.npy"))

    result = {
        "dataset_name": dataset_name,
        "method": "DB_only.py: prototype-only ablation (Contrastive+CE vs CE-only)",
        "seed": int(SEED),
        "img_dir": img_dir,
        "out_dir": root_out,
        "img_dim": int(f_l_img.shape[1]),
        "iq_dim": int(f_l_iq.shape[1]),
        "epochs": int(args.epochs),
        "lambda_cls": float(args.lambda_cls),
        "lr": float(args.lr),
        "alpha": float(ALPHA),
        "tag": args.tag,
    }

    # ==========================================================
    # 分支1：Contrastive + CE
    # ==========================================================
    if args.branch == "both":
        out_dir_con_ce = os.path.join(root_out, "contrastive_ce")
        print("\n========== Branch 1: Contrastive + CE ==========")
        iq_head_con_ce, img_head_con_ce, clf_con_ce, classes_con_ce = train_contrastive_supervised(
            f_l_iq,
            f_l_img,
            y_l,
            out_dir_con_ce,
            epochs=args.epochs,
            lambda_cls=args.lambda_cls,
            lr=args.lr
        )

        acc_con_ce_proto, pred_con_ce = fusion_classification(
            f_l_iq, f_l_img, y_l,
            f_u_iq, f_u_img, y_u,
            iq_head_con_ce, img_head_con_ce,
            name="[Contrastive+CE] Prototype Fusion"
        )

        acc_con_ce_clf, pred_con_ce_clf = classifier_evaluation(
            f_u_iq, f_u_img, y_u,
            iq_head_con_ce, img_head_con_ce, clf_con_ce, classes_con_ce,
            name="[Contrastive+CE] Classifier"
        )

        result["con_ce_proto_acc"] = float(acc_con_ce_proto)
        result["con_ce_classifier_acc"] = float(acc_con_ce_clf)
        np.save(os.path.join(out_dir_con_ce, "pred_proto_fusion.npy"),
                pred_con_ce.astype(np.int16))

    # ==========================================================
    # 分支2：CE-only 纯监督（basline 类别分类头）
    # ==========================================================
    out_dir_ce_only = os.path.join(root_out, "ce_only")
    print("\n========== Branch 2: CE-only ==========")
    iq_head_ce_only, img_head_ce_only, clf_ce_only, classes_ce_only = train_supervised_only(
        f_l_iq,
        f_l_img,
        y_l,
        out_dir_ce_only,
        epochs=args.epochs,
        lr=args.lr
    )

    acc_ce_only_proto, pred_ce_only = fusion_classification(
        f_l_iq, f_l_img, y_l,
        f_u_iq, f_u_img, y_u,
        iq_head_ce_only, img_head_ce_only,
        name="[CE-only] Prototype Fusion"
    )

    acc_ce_only_clf, pred_ce_only_clf = classifier_evaluation(
        f_u_iq, f_u_img, y_u,
        iq_head_ce_only, img_head_ce_only, clf_ce_only, classes_ce_only,
        name="[CE-only] Classifier"
    )

    result["ce_only_proto_acc"] = float(acc_ce_only_proto)
    result["ce_only_classifier_acc"] = float(acc_ce_only_clf)
    np.save(os.path.join(out_dir_ce_only, "pred_proto_fusion.npy"),
            pred_ce_only.astype(np.int16))

    # 兼容旧字段名：acc_ce_only 默认指向 CE-only 原型融合
    result["acc_ce_only"] = float(acc_ce_only_proto)

    name = f"ablation_result_{dataset_name}_seed{SEED}" + (f"_{args.tag}" if args.tag else "") + ".json"
    with open(os.path.join(root_out, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n========== SUMMARY ==========")
    if "con_ce_proto_acc" in result:
        print(f"Contrastive+CE  proto-fusion : {result['con_ce_proto_acc']:.6f}")
        print(f"Contrastive+CE  classifier   : {result['con_ce_classifier_acc']:.6f}")
    print(f"CE-only         proto-fusion : {result['ce_only_proto_acc']:.6f}")
    print(f"CE-only         classifier   : {result['ce_only_classifier_acc']:.6f}")
    print(f"FINAL_ACC={result['ce_only_proto_acc']}")
    print(f"FINAL_ACC_CE_ONLY_CLASSIFIER={result['ce_only_classifier_acc']}")
    print("Saved to:", os.path.join(root_out, name))










