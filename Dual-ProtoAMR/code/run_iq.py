import os
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from models.CNN1D import ResNet1D
from config import get_cfg, DATASET_CONFIG

DATASET_CONFIG_KEYS = sorted(DATASET_CONFIG.keys())

# ======================
# 超参
# ======================
SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
K = 5
METRIC = "cosine"
DEVICE = "cuda:5" if torch.cuda.is_available() else "cpu"


# ======================
# 数据工具
# ======================
def unify_x(x, seq_len):
    if x.ndim != 3:
        raise ValueError(f"X must be 3D, got {x.shape}")
    if x.shape[1] == 2 and x.shape[2] == seq_len:
        return x
    if x.shape[1] == seq_len and x.shape[2] == 2:
        return np.transpose(x, (0, 2, 1))
    raise ValueError(f"Unrecognized X shape: {x.shape}, expected seq_len={seq_len}")


def unify_y(y, n):
    if y.ndim != 2:
        raise ValueError(f"Y must be 2D, got {y.shape}")
    if y.shape == (n, 2):
        return y
    if y.shape == (2, n):
        return y.T
    raise ValueError(f"Unrecognized Y shape: {y.shape}")


class IQDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ======================
# 提取 ResNet1D 中间特征
# 不改模型文件，只在这里手动走到 dense 层
# ======================
@torch.no_grad()
def extract_embeddings(model, x, batch_size=256):
    model.eval()
    feats = []

    for i in tqdm(range(0, len(x), batch_size), desc="IQ Embedding"):
        batch = torch.from_numpy(x[i:i + batch_size]).float().to(DEVICE)

        # 手动前向到 dense，提取真正的中间特征
        out = model.conv1(batch.unsqueeze(dim=1)).squeeze(dim=2)
        out = model.conv2(out)
        out = model.conv3(out)
        out = model.conv4(out)
        out = model.conv5(out)
        out = model.conv6(out).view(out.size(0), -1)
        out = model.dense(out)   # 128维中间特征

        # 归一化，便于后面 cosine KNN / prototype
        out = F.normalize(out, p=2, dim=1)

        feats.append(out.cpu().numpy())

    return np.concatenate(feats, axis=0).astype(np.float32)


# ======================
# 训练
# ======================
def train_model(model, train_loader, ckpt_path):
    model.train()
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        total = 0
        correct = 0

        for xb, yb in tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}"):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            logits = model(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()

        avg_loss = total_loss / max(total, 1)
        acc = correct / max(total, 1)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}, train_acc={acc:.4f}")

    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint:", ckpt_path)


# ======================
# 主流程
# ======================
def main(dataset_name="radio512", seed=None, device=None, out_tag=None, out_dir=None):
    global SEED, DEVICE
    if seed is not None:
        SEED = int(seed)
    if device is not None:
        DEVICE = device
    if not torch.cuda.is_available():
        DEVICE = "cpu"
    print(f"[run_iq.py] dataset={dataset_name} seed={SEED} device={DEVICE} out_tag={out_tag}")

    cfg = get_cfg(dataset_name)
    x_path = cfg["x_path"]
    y_path = cfg["y_path"]
    seq_len = cfg["seq_len"]
    cnn_tag = cfg["cnn_tag"]
    data_dir = cfg["data_dir"]
    out_dir = out_dir if out_dir else cfg["iq_dir"]

    os.makedirs(out_dir, exist_ok=True)

    model = ResNet1D(dataset=cnn_tag)

    idx_l_path = os.path.join(data_dir, "idx_labeled_20.npy")
    idx_u_path = os.path.join(data_dir, "idx_unlabeled_80.npy")
    ckpt_path = os.path.join(out_dir, f"ResNet1D_20pct_encoder_{dataset_name}{'_' + out_tag if out_tag else ''}.pth")

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"Loading data for {dataset_name} ...")
    x = unify_x(np.load(x_path), seq_len)
    y = unify_y(np.load(y_path), x.shape[0]).astype(np.int32)

    if (not os.path.exists(idx_l_path)) or (not os.path.exists(idx_u_path)):
        raise FileNotFoundError(
            f"Cannot find split idx files:\n"
            f"  {idx_l_path}\n"
            f"  {idx_u_path}\n"
            f"请先运行 split.py 或 main.py --do_split 生成这些文件。"
        )

    labels = y[:, 0].astype(int)
    snrs = y[:, 1].astype(int)

    idx_l = np.load(idx_l_path).astype(np.int32)
    idx_u = np.load(idx_u_path).astype(np.int32)

    if np.intersect1d(idx_l, idx_u).size != 0:
        raise ValueError("idx_labeled and idx_unlabeled overlap! 请重新生成 split idx。")

    x_l, x_u = x[idx_l], x[idx_u]
    y_l, y_u = labels[idx_l], labels[idx_u]
    snr_u = snrs[idx_u]

    num_classes = int(len(np.unique(labels)))
    print(f"Classes={num_classes}, labeled={len(x_l)}, unlabeled={len(x_u)}, seq_len={seq_len}")
    print(f"Using ResNet1D model with dataset tag = {cnn_tag}")

    train_ds = IQDataset(x_l, y_l)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # 训练模型
    train_model(model, train_loader, ckpt_path)

    # 重新加载最优/当前模型
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    # 提取真正的中间特征
    f_l = extract_embeddings(model, x_l, BATCH_SIZE)
    f_u = extract_embeddings(model, x_u, BATCH_SIZE)

    np.save(os.path.join(out_dir, "F_labeled_iq.npy"), f_l)
    np.save(os.path.join(out_dir, "F_unlabeled_iq.npy"), f_u)

    knn = KNeighborsClassifier(n_neighbors=K, metric=METRIC, algorithm="brute")
    knn.fit(f_l, y_l)
    pred = knn.predict(f_u)

    acc = float(accuracy_score(y_u, pred))
    cm = confusion_matrix(y_u, pred)

    acc_by_snr = {}
    for s in sorted(np.unique(snr_u).tolist()):
        m = (snr_u == s)
        if m.sum() == 0:
            continue
        acc_by_snr[int(s)] = float(accuracy_score(y_u[m], pred[m]))

    print("IQ-only acc (ResNet1D feature + KNN):", acc)
    print("Acc by SNR:", acc_by_snr)

    np.save(os.path.join(out_dir, "pred_iq.npy"), pred.astype(np.int16))
    np.save(os.path.join(out_dir, "confusion_matrix_iq.npy"), cm.astype(np.int32))

    result = {
        "dataset_name": dataset_name,
        "seed": int(SEED),
        "device": DEVICE,
        "x_path": x_path,
        "y_path": y_path,
        "seq_len": int(seq_len),
        "cnn_tag": cnn_tag,
        "model_name": "ResNet1D",
        "feature_type": "dense_128_l2norm",
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "weight_decay": float(WEIGHT_DECAY),
        "knn_k": int(K),
        "metric": METRIC,
        "overall_acc": acc,
        "acc_by_snr": acc_by_snr,
        "ckpt_path": ckpt_path,
        "out_dir": out_dir,
    }

    with open(os.path.join(out_dir, "iq_train_knn_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f"iq_train_knn_results_seed{SEED}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved to:", out_dir)
    print(f"FINAL_ACC={acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alldb_radio512",
                        choices=list(DATASET_CONFIG_KEYS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out_tag", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None,
                        help="特征/结果输出目录，默认 iq_dir")
    args = parser.parse_args()
    main(args.dataset, seed=args.seed, device=args.device, out_tag=args.out_tag,
         out_dir=args.out_dir)
