import os
import numpy as np


# 原始路径
x_path = "/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainX.npy"
y_path = "/public/ly/zyt/xianyu/dataset/alldb/radio512/radio512NormTrainSnrY.npy"


# 新数据集保存路径
save_dir = "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio512"

os.makedirs(save_dir, exist_ok=True)


x_save_path = os.path.join(save_dir, "radio512AllSnrX.npy")
y_save_path = os.path.join(save_dir, "radio512AllSnrY.npy")


# =========================
# 读取原始数据
# =========================

X = np.load(x_path)
Y = np.load(y_path)


print("原始 X shape:", X.shape, X.dtype)
print("原始 Y shape:", Y.shape, Y.dtype)


# =========================
# 处理 Y
# 原始:
# Y shape = (2,N)
# 第一行=SNR
# 第二行=label
# =========================

if Y.ndim != 2:
    raise ValueError(f"Y 必须是二维, got {Y.shape}")


if Y.shape[0] == 2:
    # 转换:
    # (2,N) -> (N,2)
    Y = Y.T

elif Y.shape[1] == 2:
    pass

else:
    raise ValueError(f"无法识别 Y 形状: {Y.shape}")


# 此时:
# Y[:,0]=SNR
# Y[:,1]=label

snr = Y[:, 0].astype(np.int32)
label = Y[:, 1].astype(np.int32)


# =========================
# 统一格式
# 改为:
# 第一列 label
# 第二列 SNR
# =========================

Y_fixed = np.stack(
    [label, snr],
    axis=1
).astype(np.int32)


print("转换后 Y shape:", Y_fixed.shape)
print("转换后前5个Y:")
print(Y_fixed[:5])


# =========================
# 保留全部信噪比
# =========================

X_sub = X.astype(np.float32)
Y_sub = Y_fixed.astype(np.int32)


print("最终 X shape:", X_sub.shape)
print("最终 Y shape:", Y_sub.shape)


print(
    "SNR分布:",
    np.unique(Y_sub[:, 1], return_counts=True)
)


print(
    "Label分布:",
    np.unique(Y_sub[:, 0], return_counts=True)
)


# =========================
# 保存
# =========================

np.save(x_save_path, X_sub)
np.save(y_save_path, Y_sub)


print("保存完成:")
print(x_save_path)
print(y_save_path)