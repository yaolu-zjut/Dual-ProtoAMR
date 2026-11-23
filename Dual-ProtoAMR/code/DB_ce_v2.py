#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DB_ce_v2.py —— 三损失消融的「修正版」单脚本实现

与 v1（DB.py / DB_ce.py / DB_only.py）相比，**推理流程逐行不变**
（双域类原型 + L2 归一化余弦 + alpha 加权融合 + argmax，丢弃分类头），
只改训练侧的损失：

  修正 1：可学习温度 tau
      v1 把 TEMPERATURE 硬编码成 0.5。嵌入已被 L2 归一化，sim = cos/0.5 in [-2,2]，
      理想正交时 InfoNCE 的下界是 -ln(e^2/(e^2+127)) ≈ 2.90；实测 50 epoch 只从
      3.999 降到 3.674，几乎没有优化空间 -> 对比分支学不到东西。
      v2 用 CLIP 式 log 参数化：logits = logit_scale.exp() * sim，init tau=0.07
      （论文 Eq.(9) 自称 tau 可学习，v1 与此矛盾）。

  修正 2：同类负样本 mask（SupCon 式）
      v1 是纯实例级 InfoNCE：batch=128 / 11 类 -> 每个锚点平均有 ~10.6 个同类
      false negative 被强行推开，等于把「噪声实现」的方向编码进共享空间，
      与 CE 目标互斥。v2 在相似度矩阵上按标签 mask 掉同类非对角项。

  修正 3：三支共用同一份代码，靠 --branch 切换
      v1 是三个脚本各写一遍（DB_only 的 branch1 与 DB_ce 逐行等价，白跑一遍）。

  修正 4：直接输出按 SNR 分段的精度，便于核对论文 Table III。

分支（--branch）：
  con     : 只做跨模态对比损失          （对应 DB.py）
  con_ce  : 对比损失 + lambda * CE      （对应 DB_ce.py，论文 Eq.(15)）
  ce      : 只做 CE                     （对应 DB_only.py 的 ce_only 分支）

用法示例：
  python DB_ce_v2.py --dataset alldb_radio128 --arm v2_learnTau_supcon \
      --branch con_ce --seed 42 --device cuda:4
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

from config import get_cfg, DATASET_CONFIG

DATASET_CONFIG_KEYS = sorted(DATASET_CONFIG.keys())


# ======================
# 数据集 / 网络（与 v1 完全一致）
# ======================
class PairDataset(Dataset):
    def __init__(self, f_iq, f_img, y, snr=None):
        self.f_iq = f_iq.astype(np.float32)
        self.f_img = f_img.astype(np.float32)
        self.y = y.astype(np.int64)
        self.snr = (np.zeros(len(self.y), dtype=np.float32) if snr is None
                    else np.asarray(snr, dtype=np.float32))

    def __len__(self):
        return len(self.f_iq)

    def __getitem__(self, idx):
        return self.f_iq[idx], self.f_img[idx], self.y[idx], self.snr[idx]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class FusionClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=11):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, iq_proj, img_proj):
        z = 0.5 * iq_proj + 0.5 * img_proj
        z = F.normalize(z, dim=-1)
        return self.fc(z)


class LearnableTemp(nn.Module):
    """CLIP 式可学习温度：logits = logit_scale.exp() * sim，tau 被 clamp 到 [1/hi, 1/lo]。"""

    def __init__(self, init=0.07, tau_min=0.01, tau_max=1.0):
        super().__init__()
        self.tau_min, self.tau_max = tau_min, tau_max
        # inv_tau = exp(logit_scale)，init 时 inv_tau = 1/tau_init
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1.0 / init))))

    def inv_tau(self):
        return self.logit_scale.exp().clamp(1.0 / self.tau_max, 1.0 / self.tau_min)

    def tau(self):
        return (1.0 / self.inv_tau()).detach()


# ======================
# 损失
# ======================
def cross_modal_loss(iq_proj, img_proj, y, inv_tau, supcon=True):
    """
    对称跨模态对比损失。sim 矩阵的第 i 行第 j 列 = <z_iq[i], z_img[j]> * inv_tau。
    正样本恒为对角线；supcon=True 时把「同类但不同样本」的项从负样本里剔除。
    """
    b = iq_proj.shape[0]
    sim = torch.matmul(iq_proj, img_proj.t()) * inv_tau
    eye = torch.eye(b, dtype=torch.bool, device=sim.device)
    if supcon:
        same = y.view(-1, 1).eq(y.view(1, -1))
        keep = (~same) | eye          # 保留：不同类 + 对角线（正样本）
    else:
        keep = torch.ones_like(sim, dtype=torch.bool)
    logits = sim.masked_fill(~keep, -1e4)
    labels = torch.arange(b, device=sim.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.t(), labels)
    return (loss_a + loss_b) / 2.0


def reliability_weight(z_img, p_img, mode, w_tau, w_floor, w_gamma, snr, w_s0, w_s1,
                       w_norm="mean"):
    """
    「图像分支可靠度」逐样本权重（torch.no_grad，不参与反传）。

    物理动机：-20 ~ -6 dB 时 CLIP 图像视图几乎不含类别信息，强行跨模态对齐等于
    把噪声方向写进共享空间（实测 low8 段：纯 CE 18.96 > 对比+CE 16.06）。
    按「该样本的图像嵌入有多可信」给对齐项加权后，低 SNR 样本的权重自动趋近 0，
    整个对齐项退化为纯 CE —— 高 SNR 拿对齐的增益，低 SNR 不付对齐的代价。

    可靠度代理（均不使用标签，推理期同样可算）：
      conf   : 图像嵌入对「图像类原型」的 softmax 后验最大概率        （默认）
      margin : 原型相似度的 top1 - top2
      sim    : 原型相似度 top1（注意：实测与 SNR 负相关，只作反例）
      snr    : 直接读 SNR（oracle 上界，只作对照，实际不可用）

    w_norm 控制权重的动态范围：
      mean   : 只除以 batch 均值（原始跨度，conf 下约 1.6x，很温和）
      minmax : 先在 batch 内拉伸到 [0,1] 再 + floor（跨度可到 20x，逼近 oracle）

    返回 w，已除以 batch 均值（mean=1），只改变样本间的相对权重而不改损失量级。
    """
    with torch.no_grad():
        if mode == "snr":
            if snr is None:
                raise ValueError("--w_mode snr 需要训练集 SNR")
            w = torch.sigmoid((snr.to(z_img.device).float() - w_s0) / max(w_s1, 1e-6))
        else:
            sim = torch.matmul(z_img, p_img.t())      # 两侧都已 L2 归一化
            if mode == "sim":
                w = sim.max(dim=1).values
            elif mode == "margin":
                k = min(2, sim.shape[1])
                top = sim.topk(k, dim=1).values
                w = (top[:, 0] - top[:, 1]) if k > 1 else top[:, 0]
            else:                                     # conf
                q = F.softmax(sim / max(w_tau, 1e-6), dim=1)
                w = q.max(dim=1).values
        w = w.clamp_min(0.0)
        if w_norm == "minmax":
            lo, hi = w.min(), w.max()
            w = (w - lo) / (hi - lo).clamp_min(1e-8)
        w = w ** w_gamma + w_floor
        w = w / w.mean().clamp_min(1e-8)
    return w


def cross_modal_proto_loss(iq_proj, img_proj, y, inv_tau, weight_mode=None,
                           w_tau=0.1, w_floor=0.0, w_gamma=1.0,
                           snr=None, w_s0=-5.0, w_s1=4.0, w_norm="mean"):
    """
    类原型级跨模态对齐（--con_mode proto / proto_w）。

    把每个样本拉向「**对方模态**的同类原型」，而不是拉向它自己那一份配对样本：
        L = -log  exp(<z_iq, p_img[y_i]>/tau) / sum_c exp(<z_iq, p_img[c]>/tau)
    其中 p_img[c] = normalize(mean{ z_img[j] : y_j == c })，在当前 batch 内统计，双向对称。

    动机：实例级 InfoNCE 的正样本是「这一个样本的另一模态视图」，要把它与 115 个
    不同类样本区分开，模型必须编码该样本独有的噪声实现 —— 这与 CE 要的类级聚类
    直接冲突（实测 v1/v2 的 con_ce 训练集准确率 0.615 明显低于纯 CE 的 0.651）。
    改成原型级对齐后，对齐目标和分类目标落在同一个粒度上，两者才可能互补。

    weight_mode 非 None 时（即 --con_mode proto_w），逐样本乘上图像分支可靠度权重，
    双向都按「该样本图像嵌入的可信度」加权，低 SNR 样本自重趋 0。
    """
    dev = iq_proj.device
    classes, inv = torch.unique(y, return_inverse=True)

    def protos(z):
        p = torch.zeros(len(classes), z.shape[1], device=dev, dtype=z.dtype)
        cnt = torch.zeros(len(classes), device=dev, dtype=z.dtype)
        p.index_add_(0, inv, z)
        cnt.index_add_(0, inv, torch.ones_like(inv, dtype=z.dtype))
        return F.normalize(p / cnt.clamp(min=1).unsqueeze(1), dim=-1)

    p_img = protos(img_proj)
    p_iq = protos(iq_proj)

    logits_a = torch.matmul(iq_proj, p_img.t()) * inv_tau
    logits_b = torch.matmul(img_proj, p_iq.t()) * inv_tau

    if weight_mode is None:
        loss_a = F.cross_entropy(logits_a, inv)
        loss_b = F.cross_entropy(logits_b, inv)
        return (loss_a + loss_b) / 2.0

    w = reliability_weight(img_proj, p_img, weight_mode, w_tau, w_floor, w_gamma,
                           snr, w_s0, w_s1, w_norm=w_norm)
    la = F.cross_entropy(logits_a, inv, reduction="none")
    lb = F.cross_entropy(logits_b, inv, reduction="none")
    per_sample = 0.5 * (la + lb)
    return (w * per_sample).sum() / w.sum().clamp_min(1e-8)


def cross_modal_proto2_loss(iq_proj, img_proj, y, inv_tau):
    """
    原型对原型对齐（--con_mode proto2）：只在「类原型」这一层做跨模态对齐。

        L = -log exp(<p_iq[c], p_img[c]>/tau) / sum_c' exp(<p_iq[c], p_img[c']>/tau)

    比 proto 更激进地丢掉实例信息：低 SNR 的样本只以均值形式进入原型，
    其噪声在平均中被抵消，不会像实例级对齐那样把噪声方向写进共享空间。
    代价是每步只有 C 个原型参与，梯度更弱，所以通常要配合 CE 使用。
    """
    dev = iq_proj.device
    classes, inv = torch.unique(y, return_inverse=True)

    def protos(z):
        p = torch.zeros(len(classes), z.shape[1], device=dev, dtype=z.dtype)
        cnt = torch.zeros(len(classes), device=dev, dtype=z.dtype)
        p.index_add_(0, inv, z)
        cnt.index_add_(0, inv, torch.ones_like(inv, dtype=z.dtype))
        return F.normalize(p / cnt.clamp(min=1).unsqueeze(1), dim=-1)

    p_iq = protos(iq_proj)
    p_img = protos(img_proj)
    lab = torch.arange(len(classes), device=dev)
    sim = torch.matmul(p_iq, p_img.t()) * inv_tau
    return (F.cross_entropy(sim, lab) + F.cross_entropy(sim.t(), lab)) / 2.0


def compute_con_loss(con_mode, z_iq, z_img, yb, inv_tau, wopt=None, snr=None):
    """wopt: 可靠度权重参数字典（仅 con_mode == 'proto_w' 时生效）。"""
    if con_mode in ("proto", "proto_w"):
        o = wopt or {}
        wm = o.get("mode", "conf") if con_mode == "proto_w" else None
        return cross_modal_proto_loss(
            z_iq, z_img, yb, inv_tau, weight_mode=wm,
            w_tau=o.get("tau", 0.1), w_floor=o.get("floor", 0.0),
            w_gamma=o.get("gamma", 1.0), snr=snr,
            w_s0=o.get("s0", -5.0), w_s1=o.get("s1", 4.0),
            w_norm=o.get("norm", "mean"))
    if con_mode == "proto2":
        return cross_modal_proto2_loss(z_iq, z_img, yb, inv_tau)
    return cross_modal_loss(z_iq, z_img, yb, inv_tau, supcon=(con_mode == "supcon"))


# ======================
# 训练
# ======================
def build_label_mapping(y_l):
    y_cls = (y_l[:, 0] if y_l.ndim > 1 else y_l).astype(np.int64)
    classes = np.unique(y_cls)
    c2i = {c: i for i, c in enumerate(classes)}
    return classes, np.array([c2i[c] for c in y_cls], dtype=np.int64)


def train_branch(branch, f_l_iq, f_l_img, y_l, out_dir, args, device, snr_l=None):
    classes, y_mapped = build_label_mapping(y_l)
    num_classes = int(len(classes))

    loader = DataLoader(
        PairDataset(f_l_iq, f_l_img, y_mapped, snr=snr_l),
        batch_size=args.batch, shuffle=True, drop_last=False,
    )
    wopt = {"mode": args.w_mode, "tau": args.w_tau, "floor": args.w_floor,
            "gamma": args.w_gamma, "s0": args.w_s0, "s1": args.w_s1,
            "norm": args.w_norm}

    iq_head = ProjectionHead(input_dim=f_l_iq.shape[1]).to(device)
    img_head = ProjectionHead(input_dim=f_l_img.shape[1]).to(device)
    params = list(iq_head.parameters()) + list(img_head.parameters())

    temp = None
    if branch in ("con", "con_ce") and args.learn_tau:
        temp = LearnableTemp(args.tau_init).to(device)
        params += list(temp.parameters())

    classifier = None
    if branch in ("con_ce", "ce"):
        classifier = FusionClassifier(input_dim=64, num_classes=num_classes).to(device)
        params += list(classifier.parameters())

    opt = torch.optim.Adam(params, lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    fixed_inv_tau = 1.0 / args.tau_fixed

    for epoch in range(1, args.epochs + 1):
        iq_head.train(); img_head.train()
        if classifier is not None:
            classifier.train()

        tot = tot_con = tot_cls = 0.0
        n = 0
        correct = 0
        for iq_b, img_b, yb, sb in loader:
            iq_b = iq_b.to(device, dtype=torch.float32)
            img_b = img_b.to(device, dtype=torch.float32)
            yb = yb.to(device, dtype=torch.long)
            sb = sb.to(device, dtype=torch.float32)

            z_iq = iq_head(iq_b)
            z_img = img_head(img_b)

            loss = None
            con_v = cls_v = 0.0
            if branch in ("con", "con_ce"):
                inv_tau = temp.inv_tau() if temp is not None else fixed_inv_tau
                l_con = compute_con_loss(args.con_mode, z_iq, z_img, yb, inv_tau,
                                         wopt=wopt, snr=sb)
                loss = l_con
                con_v = float(l_con.item())
            if branch in ("con_ce", "ce"):
                logits = classifier(z_iq, z_img)
                l_cls = ce_loss(logits, yb)
                loss = l_cls if loss is None else loss + args.lambda_cls * l_cls
                cls_v = float(l_cls.item())
                correct += (logits.argmax(dim=1) == yb).sum().item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = iq_b.size(0)
            tot += float(loss.item()) * bs
            tot_con += con_v * bs
            tot_cls += cls_v * bs
            n += bs

        msg = f"[{branch}][Epoch {epoch}] loss={tot / n:.4f}"
        if branch in ("con", "con_ce"):
            msg += f", con={tot_con / n:.4f}"
        if branch in ("con_ce", "ce"):
            msg += f", cls={tot_cls / n:.4f}, train_acc={correct / n:.4f}"
        if temp is not None:
            msg += f", tau={float(temp.tau()):.4f}"
        print(msg, flush=True)

    # ---- 诊断：可靠度权重是否真的随 SNR 单调上升？用训练好的头在标注集上回算 ----
    if branch in ("con", "con_ce") and args.con_mode == "proto_w":
        iq_head.eval(); img_head.eval()
        with torch.no_grad():
            _to = lambda a: torch.from_numpy(np.asarray(a)).to(device, dtype=torch.float32)
            zI, zG = iq_head(_to(f_l_iq)), img_head(_to(f_l_img))
            yt = torch.from_numpy(y_mapped).to(device)
            cu, iv = torch.unique(yt, return_inverse=True)

            def _protos(z):
                p = torch.zeros(len(cu), z.shape[1], device=device, dtype=z.dtype)
                c = torch.zeros(len(cu), device=device, dtype=z.dtype)
                p.index_add_(0, iv, z)
                c.index_add_(0, iv, torch.ones_like(iv, dtype=z.dtype))
                return F.normalize(p / c.clamp(min=1).unsqueeze(1), dim=-1)

            wv = reliability_weight(zG, _protos(zG), args.w_mode, args.w_tau, args.w_floor,
                                    args.w_gamma, _to(snr_l), args.w_s0, args.w_s1,
                                    w_norm=args.w_norm)
        wv = wv.detach().cpu().numpy()
        sn = np.asarray(snr_l).astype(np.int64)
        print("[weight-diag] 各 SNR 上的平均可靠度权重（全标注集归一化，均值=1）：", flush=True)
        for s in sorted(set(sn.tolist())):
            m = sn == s
            print("   SNR %+3d dB  n=%5d  mean_w=%.4f" % (s, int(m.sum()), float(wv[m].mean())),
                  flush=True)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(iq_head.state_dict(), os.path.join(out_dir, "iq_proj_head.pth"))
    torch.save(img_head.state_dict(), os.path.join(out_dir, "img_proj_head.pth"))
    if classifier is not None:
        torch.save(classifier.state_dict(), os.path.join(out_dir, "fusion_classifier.pth"))
    np.save(os.path.join(out_dir, "classes.npy"), classes)

    return iq_head, img_head, classifier, classes, (float(temp.tau()) if temp is not None else None)


# ======================
# 推理（与 v1 逐行一致：双域类原型 + alpha 加权余弦）
# ======================
SEG_GROUPS = {
    "high(10~18)": [10, 12, 14, 16, 18],
    "mid(-2,0,2)": [-2, 0, 2],
    "low6(-8~2)": [-8, -6, -4, -2, 0, 2],
    "low8(-20~-6)": [-20, -18, -16, -14, -12, -10, -8, -6],
    "all": None,
}


@torch.no_grad()
def evaluate(f_l_iq, f_l_img, y_l, f_u_iq, f_u_img, y_u, snr_u,
             iq_head, img_head, classifier, alpha, device):
    iq_head.eval(); img_head.eval()
    to = lambda a: torch.from_numpy(a).to(device, dtype=torch.float32)

    zl_iq, zl_img = iq_head(to(f_l_iq)), img_head(to(f_l_img))
    zu_iq, zu_img = iq_head(to(f_u_iq)), img_head(to(f_u_img))

    y_l_cls = (y_l[:, 0] if y_l.ndim > 1 else y_l).astype(np.int64)
    y_u_cls = (y_u[:, 0] if y_u.ndim > 1 else y_u).astype(np.int64)
    classes = np.unique(y_l_cls)

    p_iq = torch.stack([zl_iq[y_l_cls == c].mean(dim=0) for c in classes])
    p_img = torch.stack([zl_img[y_l_cls == c].mean(dim=0) for c in classes])

    s_iq = F.normalize(zu_iq, dim=-1) @ F.normalize(p_iq, dim=-1).t()
    s_img = F.normalize(zu_img, dim=-1) @ F.normalize(p_img, dim=-1).t()
    s_f = alpha * s_iq + (1.0 - alpha) * s_img
    pred_proto = classes[s_f.argmax(dim=1).cpu().numpy()]

    pred_clf = None
    if classifier is not None:
        classifier.eval()
        logits = classifier(zu_iq, zu_img)
        pred_clf = classes[logits.argmax(dim=1).cpu().numpy()]

    def seg(pred, g):
        m = np.ones_like(snr_u, dtype=bool) if g is None else np.isin(snr_u, g)
        if m.sum() == 0:
            return None
        return float((pred[m] == y_u_cls[m]).mean()), int(m.sum())

    out = {"proto_overall": float((pred_proto == y_u_cls).mean()), "seg": {}}
    if pred_clf is not None:
        out["clf_overall"] = float((pred_clf == y_u_cls).mean())
    for name, g in SEG_GROUPS.items():
        r = seg(pred_proto, g)
        if not r:
            out["seg"][name] = None
            continue
        out["seg"][name] = {"n": r[1], "proto": r[0]}
        if pred_clf is not None:
            r2 = seg(pred_clf, g)
            if r2:
                out["seg"][name]["clf"] = r2[0]
    return out


# ======================
# 主流程
# ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="alldb_radio128",
                    choices=list(DATASET_CONFIG_KEYS))
    ap.add_argument("--branch", type=str, default="con_ce",
                    choices=["con", "con_ce", "ce"])
    ap.add_argument("--arm", type=str, required=True,
                    help="实验组名，如 v1_repro / v2_learnTau / v2_supcon / v2_both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:4")
    ap.add_argument("--img_dir", type=str, default=None, help="图像特征目录，默认 data_dir")
    ap.add_argument("--out_root", type=str, default=None,
                    help="默认 <base_dir>/ablation_loss_v2")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lambda_cls", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--tau_init", type=float, default=0.07, help="可学习温度的初值")
    ap.add_argument("--tau_fixed", type=float, default=0.5, help="不可学习时用的固定温度")
    ap.add_argument("--no_learn_tau", dest="learn_tau", action="store_false",
                    help="关闭可学习温度（复现 v1 行为）")
    ap.add_argument("--no_supcon", dest="supcon", action="store_false",
                    help="关闭同类负样本 mask（退回 v1 的 InfoNCE）")
    ap.add_argument("--con_mode", type=str, default=None,
                    choices=["infonce", "supcon", "proto", "proto2", "proto_w"],
                    help="对比损失形态：infonce=v1 原样；supcon=同类负样本 mask；"
                         "proto=类原型级跨模态对齐；proto_w=在 proto 上按图像分支可靠度加权。"
                         "默认由 --supcon/--no_supcon 推导。")
    # ---- proto_w 的可靠度权重参数 ----
    ap.add_argument("--w_mode", type=str, default="conf",
                    choices=["conf", "margin", "sim", "snr"],
                    help="可靠度代理：conf=图像原型后验最大概率(默认)；margin=top1-top2；"
                         "sim=top1 相似度；snr=直接用 SNR（oracle 上界，仅对照）")
    ap.add_argument("--w_tau", type=float, default=0.1, help="conf 模式 softmax 温度")
    ap.add_argument("--w_floor", type=float, default=0.0, help="权重下限（防止低 SNR 被完全归零）")
    ap.add_argument("--w_gamma", type=float, default=1.0, help="权重幂次，>1 更激进")
    ap.add_argument("--w_s0", type=float, default=-5.0, help="snr 模式 sigmoid 中心 (dB)")
    ap.add_argument("--w_s1", type=float, default=4.0, help="snr 模式 sigmoid 尺度 (dB)")
    ap.add_argument("--w_norm", type=str, default="mean", choices=["mean", "minmax"],
                    help="权重动态范围：mean=只除均值(温和)；minmax=批内拉伸到[0,1](锐利)")
    ap.set_defaults(learn_tau=True, supcon=True)
    args = ap.parse_args()

    if args.con_mode is None:
        args.con_mode = "supcon" if args.supcon else "infonce"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    cfg = get_cfg(args.dataset)
    data_dir, iq_dir, base_dir = cfg["data_dir"], cfg["iq_dir"], cfg["base_dir"]
    img_dir = args.img_dir if args.img_dir else data_dir
    out_root = args.out_root if args.out_root else os.path.join(base_dir, "ablation_loss_v2")
    out_dir = os.path.join(out_root, args.arm, args.branch)
    os.makedirs(out_dir, exist_ok=True)

    f_l_img = np.load(os.path.join(img_dir, "F_labeled.npy"))
    f_u_img = np.load(os.path.join(img_dir, "F_unlabeled.npy"))
    f_l_iq = np.load(os.path.join(iq_dir, "F_labeled_iq.npy"))
    f_u_iq = np.load(os.path.join(iq_dir, "F_unlabeled_iq.npy"))
    y_l_raw = np.load(os.path.join(data_dir, "Y_labeled_20.npy"))
    y_u_raw = np.load(os.path.join(data_dir, "Y_unlabeled_80.npy"))
    snr_u = y_u_raw[:, 1].astype(np.int64) if y_u_raw.ndim > 1 else np.zeros(len(y_u_raw), np.int64)
    snr_l = y_l_raw[:, 1].astype(np.int64) if y_l_raw.ndim > 1 else np.zeros(len(y_l_raw), np.int64)

    print(f"[DB_ce_v2] dataset={args.dataset} arm={args.arm} branch={args.branch} "
          f"seed={args.seed} device={device} learn_tau={args.learn_tau} con_mode={args.con_mode} "
          f"w_mode={args.w_mode} w_tau={args.w_tau} w_floor={args.w_floor} w_gamma={args.w_gamma} "
          f"w_norm={args.w_norm} "
          f"tau_init={args.tau_init} tau_fixed={args.tau_fixed} epochs={args.epochs} "
          f"img_dim={f_l_img.shape[1]} "
          f"iq_dim={f_l_iq.shape[1]}", flush=True)

    iq_head, img_head, classifier, classes, final_tau = train_branch(
        args.branch, f_l_iq, f_l_img, y_l_raw, out_dir, args, device, snr_l=snr_l)

    res = evaluate(f_l_iq, f_l_img, y_l_raw, f_u_iq, f_u_img, y_u_raw, snr_u,
                   iq_head, img_head, classifier, args.alpha, device)

    result = {
        "script": "DB_ce_v2.py",
        "dataset": args.dataset,
        "arm": args.arm,
        "branch": args.branch,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "batch": int(args.batch),
        "lr": float(args.lr),
        "lambda_cls": float(args.lambda_cls),
        "alpha": float(args.alpha),
        "learn_tau": bool(args.learn_tau),
        "supcon": bool(args.supcon),
        "con_mode": args.con_mode,
        "w_mode": args.w_mode,
        "w_tau": float(args.w_tau),
        "w_floor": float(args.w_floor),
        "w_gamma": float(args.w_gamma),
        "w_norm": args.w_norm,
        "tau_init": float(args.tau_init),
        "tau_fixed": float(args.tau_fixed),
        "final_tau": final_tau,
        "img_dim": int(f_l_img.shape[1]),
        "iq_dim": int(f_l_iq.shape[1]),
        "num_classes": int(len(classes)),
        "N_unlabeled": int(len(y_u_raw)),
        "overall_acc": float(res["proto_overall"]),
        "clf_overall_acc": res.get("clf_overall"),
        "seg": res["seg"],
        "heads": {
            "iq": os.path.join(out_dir, "iq_proj_head.pth"),
            "img": os.path.join(out_dir, "img_proj_head.pth"),
            "clf": (os.path.join(out_dir, "fusion_classifier.pth") if classifier is not None else None),
        },
    }
    name = f"v2result_{args.dataset}_{args.arm}_{args.branch}_seed{args.seed}.json"
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Fusion Accuracy:", res["proto_overall"], flush=True)
    print(f"FINAL_ACC={float(res['proto_overall'])}", flush=True)
    print("Saved:", os.path.join(out_dir, name), flush=True)


if __name__ == "__main__":
    main()
