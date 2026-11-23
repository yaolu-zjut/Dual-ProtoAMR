# Dual-ProtoAMR

Semi-supervised automatic modulation recognition with **dual-domain prototypes**:
a frozen CLIP vision encoder reads the *picture* of an IQ segment, a ResNet1D reads
its *waveform*, and both are aligned by cross-modal contrastive learning.
Only **20 %** of the samples are labeled; the remaining **80 %** are classified by
cosine similarity to per-class prototypes.

Final setting used for the paper:

| item | value |
|---|---|
| image branch | 4 views (constellation + 2/4/8-order spectrum) → CLIP ViT-B/32 → 2048-d |
| signal branch | ResNet1D on raw IQ → 128-d |
| projection heads | 64-d, L2-normalized |
| inference | class prototypes, `s = 0.5·s_iq + 0.5·s_img`, argmax |
| split | 20 % labeled / 80 % unlabeled, stratified by (class, SNR) |
| epochs / seed | 50 / 42 |

All paper code is in [`code/`](code/). Nothing was refactored — these are the scripts
that produced the reported numbers.

---

## 1. Install

```bash
python -m pip install -r requirements.txt
```

Tested with Python 3.8 + CUDA. Install the matching PyTorch build from
<https://pytorch.org> first if you need a specific CUDA version.

> `transformers` must stay on **4.x**. `code/run.py` uses the tensor returned by
> `get_image_features()`; 5.x returns an output object instead and will crash.

## 2. Prepare the data

Each dataset is two NumPy files:

| file | shape | dtype | content |
|---|---|---|---|
| `X.npy` | `(N, L, 2)` or `(N, 2, L)` | float32 | I/Q samples, `L` = 128 / 512 / 1024 |
| `Y.npy` | `(N, 2)` or `(2, N)` | int | `[label, snr]` |

Then open [`code/config.py`](code/config.py) and edit:

```python
ROOT = "/path/to/your/workspace"          # all intermediate features go here
DATASET_CONFIG = {
    "radio128": {
        "x_path": "/path/to/X.npy",
        "y_path": "/path/to/Y.npy",
        "seq_len": 128,
        "cnn_tag": "128",
    },
}
```

`config.py` ships with the absolute paths of our cluster — **you must replace them.**
RML2016.10a can be converted into this format with `code/allsnr.py` (it dumps
`radio128AllSnrX.npy` / `radio128AllSnrY.npy`).

## 3. Prepare the CLIP weights

The image branch loads CLIP **from a local directory only**, downloads are disabled
(`local_files_only=True`). Put the checkpoint somewhere and point
[`code/run.py`](code/run.py) line 35 at it:

```python
MODEL_NAME = "/path/to/clip-vit-base-patch32"
```

The folder must contain `config.json`, `model.safetensors`, `preprocessor_config.json`
and the tokenizer files (`openai/clip-vit-base-patch32`, ~580 MB).
Weights are **not** included in this repository.

## 4. Run — final setting

```bash
cd code

# 1) 20/80 stratified split  ->  <ROOT>/<dataset>/data/
python split.py --dataset radio128

# 2) image branch, four views -> <ROOT>/<dataset>/data/F_labeled.npy (2048-d)
python run.py --dataset radio128 --view four --device cuda:0

# 3) signal branch            -> <ROOT>/<dataset>/data_iq/F_labeled_iq.npy (128-d)
python run_iq.py --dataset radio128 --device cuda:0

# 4) main method: contrastive + 64*CE, prototype fusion, alpha = 0.5
python DB_ce_tune.py --dataset radio128 \
    --lambda_cls 64 --train_alpha 0.5 --temp 0.5 \
    --epochs 50 --seed 42 --device cuda:0
```

The last step prints and stores the accuracy:

```
FINAL_ACC=0.xxxx
Saved to: <ROOT>/<dataset>/contrastive_heads_ce/tune_radio128_seed42/fusion_result_radio128_seed42.json
```

`--device` is mandatory in practice: the scripts default to a fixed GPU id from our
cluster. Check `nvidia-smi` and pick a free one.

> **Full-SNR 128 dataset.** `split.py` only whitelists `radio128 / radio512 / radio1024`
> on the command line. To use another key defined in `config.py` (e.g. `alldb_radio128`,
> 11 classes × 20 SNR levels), call it directly — the other scripts already accept any key:
>
> ```bash
> python -c "import split; split.main('alldb_radio128')"
> python run.py --dataset alldb_radio128 --view four --device cuda:0
> python run_iq.py --dataset alldb_radio128 --device cuda:0
> python DB_ce_tune.py --dataset alldb_radio128 --lambda_cls 64 --epochs 50 --device cuda:0
> ```

## 5. Outputs

```
<ROOT>/<dataset>/
├── data/                    idx_*.npy, X/Y_labeled_20.npy, X/Y_unlabeled_80.npy,
│                            F_labeled.npy, F_unlabeled.npy      (image features)
├── data_iq/                 F_labeled_iq.npy, F_unlabeled_iq.npy (signal features)
└── contrastive_heads_ce/
    └── tune_<tag>/          projection heads (.pth) + fusion_result_<tag>.json
```

The headline number is `overall_acc` in the JSON (unlabeled 80 %).

## 6. Baselines and ablations

| script | command | reports |
|---|---|---|
| `run.py` | `--view four` / `constellation` / `spectrum2` / `spectrum4` / `spectrum8` | CLIP + kNN, and the per-view ablation |
| `run_iq.py` | `--dataset ...` | ResNet1D + kNN (signal branch only) |
| `run_fusion.py` | `--dataset ...` | training-free prototype fusion baseline |
| `DB.py` | `--dataset ...` | InfoNCE only, prototype inference (no CE) |
| `DB_ce.py` | `--dataset ... --lambda_cls 64` | InfoNCE + CE, prototype **and** classifier head |
| `DB_ce_proto.py` | `--dataset ... --lambda_cls 30 --tau 0.1` | prototypical loss instead of the CE head |
| `DB_only.py` | `--dataset ... --branch ce_only` | CE-only ablation (no contrastive term) |
| `DB_ce_v2.py` | `--arm v2_learnTau_supcon --branch con_ce` | exploratory fix (learnable τ + SupCon mask), *not* used for the main results |

`DB_ce_tune.py` and `DB_ce.py` share the same training/inference maths;
`DB_ce_tune.py` just exposes `λ_cls / train_alpha / temp / epochs` on the command line
and writes into its own `tune_*/` folder.

## 7. Notes

- Constellation images are saved with `bbox_inches="tight"`, so their pixel width
  varies slightly from sample to sample. This is the original behaviour and the
  reported numbers were produced with it — leave it as is if you want to reproduce them.
- `run.py` renders ~0.04 s per sample single-threaded (≈2.3 h for 220 k samples).
