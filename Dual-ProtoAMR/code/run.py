import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from transformers import CLIPProcessor, CLIPModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from config import get_cfg, DATASET_CONFIG

DATASET_CONFIG_KEYS = sorted(DATASET_CONFIG.keys())


# ======================
# 超参
# ======================
SEED = 42
IMG_SIZE = 224
DPI = 224
FIG_INCH = IMG_SIZE / DPI
SCATTER_S = 8
LINE_W = 1.5
SPECTRUM_NFFT = 512
SPECTRA_ORDERS = (2, 4, 8)
MODEL_NAME = "/public/mount_data/data/yaolu/signal_group/ghy/zero_short/clip-vit-base-patch32"
BATCH_SIZE = 256
K = 5
METRIC = "cosine"
MAX_PER_CLASS = None
VIEW = "constellation"

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


# ======================
# IQ -> PIL 图像
# ======================
def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def constellation_pil(i_data, q_data):
    fig = plt.figure(figsize=(FIG_INCH, FIG_INCH), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.scatter(i_data, q_data, s=SCATTER_S)
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")
    return fig_to_pil(fig)


def spectrum_pil(z, p):
    y = z ** p
    y_fft = np.fft.fftshift(np.fft.fft(y, n=SPECTRUM_NFFT))
    mag = np.abs(y_fft)
    mag = mag / (mag.max() + 1e-12)
    mag_db = 20 * np.log10(mag + 1e-12)
    freq = np.linspace(-0.5, 0.5, SPECTRUM_NFFT, endpoint=False)

    fig = plt.figure(figsize=(FIG_INCH, FIG_INCH), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.plot(freq, mag_db, linewidth=LINE_W)
    ax.set_axis_off()
    ax.set_xlim(-0.5, 0.5)
    return fig_to_pil(fig)


# def sample_to_pils(x_2_l):
#     i = x_2_l[0]
#     q = x_2_l[1]
#     z = i + 1j * q
#
#     imgs = [constellation_pil(i, q)]
#     for p in SPECTRA_ORDERS:
#         imgs.append(spectrum_pil(z, p))
#     return imgs
def sample_to_pils(x_2_l):
    i = x_2_l[0]
    q = x_2_l[1]
    z = i + 1j * q

    if VIEW == "four":
        imgs = [constellation_pil(i, q)]
        for p in SPECTRA_ORDERS:
            imgs.append(spectrum_pil(z, p))
        return imgs

    elif VIEW == "constellation":
        return [constellation_pil(i, q)]

    elif VIEW == "spectrum2":
        return [spectrum_pil(z, 2)]

    elif VIEW == "spectrum4":
        return [spectrum_pil(z, 4)]

    elif VIEW == "spectrum8":
        return [spectrum_pil(z, 8)]

    else:
        raise ValueError(f"Unknown VIEW: {VIEW}")


# ======================
# CLIP embedding
# ======================
def load_clip(model_name=MODEL_NAME, device=None):
    if device is None:
        device = "cuda:6" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device).eval()
    proc = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    return device, model, proc


@torch.no_grad()
def encode_pil_batch(pil_images, device, model, proc):
    inputs = proc(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    feats = model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats.detach().cpu().numpy().astype(np.float32)


def encode_dataset(x, device, model, proc, batch_size=64):
    """
    x: (N,2,seq_len)
    return: embeddings (N, D_total)
    """
    n_total = x.shape[0]
    all_embeds = []

    tmp_imgs = sample_to_pils(x[0])
    tmp = encode_pil_batch(tmp_imgs, device, model, proc)
    d = tmp.shape[1]
    m = len(tmp_imgs)
    d_total = d * m

    for start in tqdm(range(0, n_total, batch_size), desc="Embedding"):
        end = min(n_total, start + batch_size)
        pil_list = []
        for i in range(start, end):
            pil_list.extend(sample_to_pils(x[i]))

        feats = encode_pil_batch(pil_list, device, model, proc)
        feats = feats.reshape((end - start, m * d))
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        all_embeds.append(feats.astype(np.float32))

    emb = np.concatenate(all_embeds, axis=0)
    assert emb.shape == (n_total, d_total)
    return emb


# ======================
# kNN + 评估
# ======================
def eval_knn(f_l, y_l, f_u, y_u, snr_u, k=5):
    knn = KNeighborsClassifier(n_neighbors=k, metric=METRIC, algorithm="brute")
    knn.fit(f_l, y_l)
    pred = knn.predict(f_u)

    acc = float(accuracy_score(y_u, pred))
    cm = confusion_matrix(y_u, pred)

    acc_by_snr = {}
    snr_u = snr_u.astype(int)
    for s in sorted(np.unique(snr_u).tolist()):
        mask = (snr_u == s)
        if mask.sum() == 0:
            continue
        acc_by_snr[int(s)] = float(accuracy_score(y_u[mask], pred[mask]))

    return acc, cm, acc_by_snr, pred


# ======================
# 主流程
# ======================
def main(dataset_name="radio128", view=None, img_dir=None, device=None):
    global VIEW

    if view is not None:
        VIEW = view
    if VIEW not in ("four", "constellation", "spectrum2", "spectrum4", "spectrum8"):
        raise ValueError(f"Unknown VIEW: {VIEW}")

    cfg = get_cfg(dataset_name)
    x_path = cfg["x_path"]
    y_path = cfg["y_path"]
    seq_len = cfg["seq_len"]
    data_dir = cfg["data_dir"]          # 切分 idx / 标签固定放这里
    out_dir = img_dir if img_dir else data_dir   # 特征输出目录（按视图隔离）

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[run.py] dataset={dataset_name} view={VIEW} img_dir={out_dir} device={device}")

    idx_labeled_path = os.path.join(data_dir, "idx_labeled_20.npy")
    idx_unlabeled_path = os.path.join(data_dir, "idx_unlabeled_80.npy")

    print(f"Loading data for {dataset_name} ...")
    x = np.load(x_path)
    y = np.load(y_path)

    x = unify_x(x, seq_len)
    y = unify_y(y, x.shape[0]).astype(np.int32)

    if MAX_PER_CLASS is not None:
        raise ValueError("MAX_PER_CLASS must be None when using pre-saved idx split.")

    labels = y[:, 0]
    snrs = y[:, 1]
    n_total = x.shape[0]
    print(f"X: {x.shape}  Y: {y.shape}  N={n_total}")
    print(f"Unique labels: {len(np.unique(labels))}, unique snrs: {len(np.unique(snrs))}")

    if (not os.path.exists(idx_labeled_path)) or (not os.path.exists(idx_unlabeled_path)):
        raise FileNotFoundError(
            f"Cannot find split idx files:\n"
            f"  {idx_labeled_path}\n"
            f"  {idx_unlabeled_path}\n"
            f"请先运行 split.py 或 main.py --do_split 生成这些文件。"
        )

    labeled_idx = np.load(idx_labeled_path).astype(np.int32)
    unlabeled_idx = np.load(idx_unlabeled_path).astype(np.int32)

    if np.intersect1d(labeled_idx, unlabeled_idx).size != 0:
        raise ValueError("idx_labeled and idx_unlabeled overlap! 请重新生成 split idx。")
    if labeled_idx.max() >= n_total or unlabeled_idx.max() >= n_total:
        raise ValueError("idx contains out-of-range indices! 请确认 split idx 对应同一份数据集。")

    x_l, y_l_full = x[labeled_idx], y[labeled_idx]
    x_u, y_u_full = x[unlabeled_idx], y[unlabeled_idx]
    print(f"Split loaded: labeled={len(labeled_idx)} unlabeled={len(unlabeled_idx)}")

    # np.save(os.path.join(out_dir, "X_labeled_20.npy"), x_l.astype(np.float32))
    # np.save(os.path.join(out_dir, "Y_labeled_20.npy"), y_l_full.astype(np.int16))
    # np.save(os.path.join(out_dir, "X_unlabeled_80.npy"), x_u.astype(np.float32))
    # np.save(os.path.join(out_dir, "Y_unlabeled_80.npy"), y_u_full.astype(np.int16))

    device, model, proc = load_clip(MODEL_NAME, device=device)
    print(f"Encoding with {MODEL_NAME} on {device} (view={VIEW}) ...")

    f_l = encode_dataset(x_l, device, model, proc, batch_size=BATCH_SIZE)
    f_u = encode_dataset(x_u, device, model, proc, batch_size=BATCH_SIZE)

    np.save(os.path.join(out_dir, "F_labeled.npy"), f_l)
    np.save(os.path.join(out_dir, "F_unlabeled.npy"), f_u)

    y_l = y_l_full[:, 0].astype(int)
    y_u = y_u_full[:, 0].astype(int)
    snr_u = y_u_full[:, 1].astype(int)

    acc, cm, acc_by_snr, pred = eval_knn(f_l, y_l, f_u, y_u, snr_u, k=K)

    np.save(os.path.join(out_dir, "pred_unlabeled.npy"), pred.astype(np.int16))
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm.astype(np.int32))

    result = {
        "dataset_name": dataset_name,
        "view": VIEW,
        "img_dir": out_dir,
        "data_dir": data_dir,
        "x_path": x_path,
        "y_path": y_path,
        "seq_len": int(seq_len),
        "model": MODEL_NAME,
        "spectra_orders": list(SPECTRA_ORDERS),
        "seed": SEED,
        "knn_k": K,
        "metric": METRIC,
        "batch_size": BATCH_SIZE,
        "N_total": int(n_total),
        "N_labeled": int(len(labeled_idx)),
        "N_unlabeled": int(len(unlabeled_idx)),
        "embedding_dim": int(f_l.shape[1]),
        "overall_acc": float(acc),
        "acc_by_snr": acc_by_snr,
        "split_idx_paths": {
            "idx_labeled": idx_labeled_path,
            "idx_unlabeled": idx_unlabeled_path,
        },
    }
    with open(os.path.join(out_dir, "knn_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("========== DONE ==========")
    print("View:", VIEW)
    print("Overall acc:", acc)
    print("Acc by SNR:", acc_by_snr)
    print("Saved to:", out_dir)
    print(f"FINAL_ACC={acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="alldb_radio512",
                        choices=list(DATASET_CONFIG_KEYS))
    parser.add_argument("--view", type=str, default=VIEW,
                        choices=["four", "constellation", "spectrum2", "spectrum4", "spectrum8"])
    parser.add_argument("--img_dir", type=str, default=None,
                        help="特征输出目录；不传则写到 data_dir（保持旧行为）")
    parser.add_argument("--device", type=str, default=None, help="如 cuda:2；不传则用默认 cuda:6")
    args = parser.parse_args()
    main(args.dataset, view=args.view, img_dir=args.img_dir, device=args.device)
