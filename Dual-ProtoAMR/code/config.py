import os

ROOT = "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2"

DATASET_CONFIG = {
    "radio128": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio128/radio128NormTrainSnrY.npy",
        "seq_len": 128,
        "cnn_tag": "128",
    },
    "radio512": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio512/radio512NormTrainSnrY.npy",
        "seq_len": 512,
        "cnn_tag": "512",
    },

    "radio1024": {  # 新添加的配置
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,  # 设置合适的 seq_len
        "cnn_tag": "1024",  # 设置合适的 cnn_tag
    },

    "radio128_lowsnr1": {
    "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio128_lowsnr1/radio128LowSnrX.npy",
    "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio128_lowsnr1/radio128LowSnrSnrY.npy",
    "seq_len": 128,
    "cnn_tag": "128",
    },

    "radio128_lowsnr2": {
    "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio128_lowsnr2/radio128LowSnrX.npy",
    "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio128_lowsnr2/radio128LowSnrSnrY.npy",
    "seq_len": 128,
    "cnn_tag": "128",
    },

    "radio512_lowsnr1": {
    "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio512_lowsnr1/radio512LowSnrX.npy",
    "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio512_lowsnr1/radio512LowSnrSnrY.npy",
    "seq_len": 512,
    "cnn_tag": "512",
},
    "radio512_lowsnr2": {
    "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio512_lowsnr2/radio512LowSnrX.npy",
    "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/lowsnr_data/radio512_lowsnr2/radio512LowSnrSnrY.npy",
    "seq_len": 512,
    "cnn_tag": "512",
},
    "alldb_radio512": {
        "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio512/radio512AllSnrX.npy",
        "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio512/radio512AllSnrY.npy",
        "seq_len": 512,
        "cnn_tag": "512",
    },
    "alldb_radio128": {
        "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio128/radio128AllSnrX.npy",
        "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio128/radio128AllSnrY.npy",
        "seq_len": 128,
        "cnn_tag": "128",
    },
    "alldb_radio128_new": {
        "x_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio128/radio128AllSnrX.npy",
        "y_path": "/public/mount_data/data/yaolu/signal_group/ghy/zero_short2/data/alldb_radio128/radio128AllSnrY.npy",
        "seq_len": 128,
        "cnn_tag": "128",
    },
    "radio1024_s2023": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024_s2024": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024_s2025": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_constellation_s2023": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum2_s2023": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum4_s2023": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum8_s2023": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_constellation_s2024": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum2_s2024": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum4_s2024": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum8_s2024": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_constellation_s2025": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum2_s2025": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum4_s2025": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },
    "radio1024sv_spectrum8_s2025": {
        "x_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainX.npy",
        "y_path": "/public/ly/zyt/xianyu/dataset/radio1024/radio1024NormTrainSnrY.npy",
        "seq_len": 1024,
        "cnn_tag": "1024",
    },


}

def get_cfg(dataset_name: str):
    base = DATASET_CONFIG[dataset_name].copy()
    base["dataset_name"] = dataset_name
    base["base_dir"] = os.path.join(ROOT, dataset_name)
    base["data_dir"] = os.path.join(base["base_dir"], "data")
    base["img_dir"] = os.path.join(base["base_dir"], "data_img")
    base["iq_dir"] = os.path.join(base["base_dir"], "data_iq")
    base["fusion_dir"] = os.path.join(base["base_dir"], "data_fusion")
    return base
