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
DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
TEMPERATURE = 0.5
ALPHA = 0.5  # 融合权重：IQ主导


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ======================
# 数据集
# ======================
class IQImgContrastDataset(Dataset):
    def __init__(self, f_iq, f_img):
        self.f_iq = f_iq.astype(np.float32)
        self.f_img = f_img.astype(np.float32)
    def __len__(self):
        return len(self.f_iq)
    def __getitem__(self, idx):
        return self.f_iq[idx], self.f_img[idx]

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
def train_contrastive(f_l_iq, f_l_img, out_dir, epochs=30):
    os.makedirs(out_dir, exist_ok=True)
    dataset = IQImgContrastDataset(f_l_iq, f_l_img)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    iq_head = ProjectionHead(input_dim=f_l_iq.shape[1]).to(DEVICE)
    img_head = ProjectionHead(input_dim=f_l_img.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(list(iq_head.parameters()) + list(img_head.parameters()), lr=1e-3)

    for epoch in range(1, epochs+1):
        total_loss = 0.0
        for iq_batch, img_batch in loader:
            iq_batch = iq_batch.to(DEVICE, dtype=torch.float32)
            img_batch = img_batch.to(DEVICE, dtype=torch.float32)
            iq_proj = iq_head(iq_batch)
            img_proj = img_head(img_batch)
            loss = contrastive_loss(iq_proj, img_proj)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * iq_batch.size(0)
        print(f"[Epoch {epoch}] Loss={total_loss/len(dataset):.4f}")

    torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head.pth"))
    torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head.pth"))
    print("Projection heads saved.")
    return iq_head, img_head

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
                        help="投影头/结果输出目录，默认 base_dir/contrastive_heads")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--temp", type=float, default=TEMPERATURE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    SEED = args.seed
    if args.device:
        DEVICE = args.device
    if not torch.cuda.is_available():
        DEVICE = "cpu"
    TEMPERATURE = args.temp
    ALPHA = args.alpha
    set_seed(SEED)

    dataset_name = args.dataset
    cfg = get_cfg(dataset_name)

    data_dir = cfg["data_dir"]
    iq_dir = args.iq_dir if args.iq_dir else cfg["iq_dir"]
    base_dir = cfg["base_dir"]

    img_dir = args.img_dir if args.img_dir else data_dir
    out_dir = args.out_dir if args.out_dir else os.path.join(base_dir, "contrastive_heads")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[DB.py] dataset={dataset_name} seed={SEED} device={DEVICE} "
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

    # 对比学习训练
    iq_head, img_head = train_contrastive(f_l_iq, f_l_img, out_dir, epochs=args.epochs)

    # 融合分类
    acc, y_pred = fusion_classification(
        f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, iq_head, img_head
    )

    np.save(os.path.join(out_dir, "pred_proto_fusion.npy"), y_pred.astype(np.int16))

    result = {
        "dataset_name": dataset_name,
        "method": "DB.py: cross-modal contrastive loss + prototype fusion",
        "seed": int(SEED),
        "img_dir": img_dir,
        "out_dir": out_dir,
        "img_dim": int(f_l_img.shape[1]),
        "iq_dim": int(f_l_iq.shape[1]),
        "epochs": int(args.epochs),
        "temperature": float(TEMPERATURE),
        "alpha": float(ALPHA),
        "overall_acc": float(acc),
        "tag": args.tag,
    }
    name = f"fusion_result_{dataset_name}_seed{SEED}" + (f"_{args.tag}" if args.tag else "") + ".json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"FINAL_ACC={float(acc)}")
    print("Saved to:", os.path.join(out_dir, name))
