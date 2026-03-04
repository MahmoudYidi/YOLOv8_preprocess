import os, json, glob
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.neighbors import NearestNeighbors
import joblib


# -------------------------
# USER SETTINGS
# -------------------------
DATA_ROOT = "data"                 # change this
OUT_DIR = "patchcore_out"          # change this if you want

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESIZE = 256
CROP = 224
EPS = 1e-6

BATCH_SIZE = 4
NUM_WORKERS = 0

# Choose third channel:
# False -> ratio = LE_log / HE_log
# True  -> contrast = LE_log - HE_log
USE_CONTRAST = False

# Keep it light (subsample memory bank patches)
# Set 0 to keep all patches (not recommended if images are large)
MEMORY_SUBSAMPLE = 50000


# -------------------------
# Helpers
# -------------------------
def list_npy(root_dir):
    files = sorted(glob.glob(os.path.join(root_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {root_dir}")
    return files


def load_le_he_npy(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3D array, got shape {arr.shape}")

    if arr.shape[-1] == 2:      # (H,W,2)
        le, he = arr[..., 0], arr[..., 1]
    elif arr.shape[0] == 2:     # (2,H,W)
        le, he = arr[0], arr[1]
    else:
        raise ValueError(f"{path}: expected (H,W,2) or (2,H,W), got {arr.shape}")

    return le, he


def preprocess(le, he, eps=EPS, resize=RESIZE, crop=CROP, use_contrast=USE_CONTRAST):
    """
    Output: (3, crop, crop) float32 = [LE_log, HE_log, ratio_or_contrast]
    """
    # scale to [0,1] only if values look like raw counts
    le_max = float(le.max()) if le.size else 1.0
    he_max = float(he.max()) if he.size else 1.0
    if le_max > 1.5:
        le = le / max(le_max, 1.0)
    if he_max > 1.5:
        he = he / max(he_max, 1.0)

    le_log = -np.log(le + eps)
    he_log = -np.log(he + eps)

    third = (le_log - he_log) if use_contrast else (le_log / (he_log + eps))

    x = np.stack([le_log, he_log, third], axis=0).astype(np.float32)  # (3,H,W)

    xt = torch.from_numpy(x).unsqueeze(0)  # (1,3,H,W)
    xt = F.interpolate(xt, size=(resize, resize), mode="bilinear", align_corners=False)

    if crop is not None:
        _, _, H, W = xt.shape
        y0 = (H - crop) // 2
        x0 = (W - crop) // 2
        xt = xt[:, :, y0:y0 + crop, x0:x0 + crop]

    return xt.squeeze(0).numpy().astype(np.float32)


def compute_dataset_stats(train_files):
    """
    Dataset-level per-channel mean/std over ALL pixels of ALL training NORMAL images.
    """
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    count = 0

    for p in tqdm(train_files, desc="Computing dataset mean/std"):
        le, he = load_le_he_npy(p)
        x = preprocess(le, he)
        x_flat = x.reshape(3, -1).astype(np.float64)
        sum_c += x_flat.sum(axis=1)
        sumsq_c += (x_flat ** 2).sum(axis=1)
        count += x_flat.shape[1]

    mean = sum_c / max(count, 1)
    var = (sumsq_c / max(count, 1)) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class DualEnergyDataset(Dataset):
    def __init__(self, files, mean, std):
        self.files = files
        self.mean = mean.reshape(3, 1, 1).astype(np.float32)
        self.std = std.reshape(3, 1, 1).astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        le, he = load_le_he_npy(p)
        x = preprocess(le, he)
        x = (x - self.mean) / (self.std + 1e-6)
        name = os.path.splitext(os.path.basename(p))[0]
        return torch.from_numpy(x).float(), name


class ResNet18Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def build_patch_embeddings(f2, f3):
    B, C2, H2, W2 = f2.shape
    f3_up = F.interpolate(f3, size=(H2, W2), mode="bilinear", align_corners=False)
    return torch.cat([f2, f3_up], dim=1)  # (B,C,H2,W2)


def main():
    train_dir = os.path.join(DATA_ROOT, "train", "normal")
    train_files = list_npy(train_dir)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) dataset stats
    mean, std = compute_dataset_stats(train_files)
    with open(os.path.join(OUT_DIR, "stats.json"), "w") as f:
        json.dump({
            "mean": mean.tolist(),
            "std": std.tolist(),
            "resize": RESIZE,
            "crop": CROP,
            "eps": EPS,
            "use_contrast": USE_CONTRAST,
        }, f, indent=2)

    # 2) dataset + loader
    ds = DualEnergyDataset(train_files, mean, std)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 3) backbone
    backbone = ResNet18Features().to(DEVICE).eval()

    # 4) memory bank
    memory = []
    with torch.no_grad():
        for x, _ in tqdm(dl, desc="Extracting train features"):
            x = x.to(DEVICE)
            f2, f3 = backbone(x)
            emb = build_patch_embeddings(f2, f3)  # (B,C,h,w)
            patches = emb.permute(0, 2, 3, 1).reshape(-1, emb.shape[1])
            memory.append(patches.cpu().numpy().astype(np.float32))

    memory = np.concatenate(memory, axis=0)
    print("Memory bank patches:", memory.shape)

    if MEMORY_SUBSAMPLE and MEMORY_SUBSAMPLE > 0 and memory.shape[0] > MEMORY_SUBSAMPLE:
        idx = np.random.choice(memory.shape[0], size=MEMORY_SUBSAMPLE, replace=False)
        memory = memory[idx]
        print("Subsampled memory bank:", memory.shape)

    np.save(os.path.join(OUT_DIR, "memory.npy"), memory)

    # 5) kNN
    knn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    knn.fit(memory)
    joblib.dump(knn, os.path.join(OUT_DIR, "knn.joblib"))

    print("Saved PatchCore artifacts to:", OUT_DIR)


if __name__ == "__main__":
    main()


import os, json, glob
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import models

import joblib


# -------------------------
# USER SETTINGS
# -------------------------
DATA_ROOT = "data"                   # change this
MODEL_DIR = "patchcore_out"          # output from training
OUT_DIR = "patchcore_test_out"       # change this

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.55                         # overlay opacity

# image-level score choice:
# "max" or "topk"
SCORE_MODE = "max"
TOPK_FRAC = 0.01                     # top 1% if SCORE_MODE="topk"


# -------------------------
# Helpers
# -------------------------
def list_npy(root_dir):
    return sorted(glob.glob(os.path.join(root_dir, "*.npy")))


def load_le_he_npy(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{path}: expected 3D array, got shape {arr.shape}")

    if arr.shape[-1] == 2:
        le, he = arr[..., 0], arr[..., 1]
    elif arr.shape[0] == 2:
        le, he = arr[0], arr[1]
    else:
        raise ValueError(f"{path}: expected (H,W,2) or (2,H,W), got {arr.shape}")

    return le, he


def preprocess(le, he, mean, std, eps, resize, crop, use_contrast):
    # scale to [0,1] if values look like raw counts
    le_max = float(le.max()) if le.size else 1.0
    he_max = float(he.max()) if he.size else 1.0
    if le_max > 1.5:
        le = le / max(le_max, 1.0)
    if he_max > 1.5:
        he = he / max(he_max, 1.0)

    le_log = -np.log(le + eps)
    he_log = -np.log(he + eps)
    third = (le_log - he_log) if use_contrast else (le_log / (he_log + eps))

    x = np.stack([le_log, he_log, third], axis=0).astype(np.float32)

    xt = torch.from_numpy(x).unsqueeze(0)
    xt = F.interpolate(xt, size=(resize, resize), mode="bilinear", align_corners=False)

    if crop is not None:
        _, _, H, W = xt.shape
        y0 = (H - crop) // 2
        x0 = (W - crop) // 2
        xt = xt[:, :, y0:y0 + crop, x0:x0 + crop]

    x = xt.squeeze(0).numpy()
    x = (x - mean.reshape(3, 1, 1)) / (std.reshape(3, 1, 1) + 1e-6)
    return x  # (3,crop,crop)


class ResNet18Features(torch.nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        return f2, f3


def build_patch_embeddings(f2, f3):
    B, C2, H2, W2 = f2.shape
    f3_up = F.interpolate(f3, size=(H2, W2), mode="bilinear", align_corners=False)
    return torch.cat([f2, f3_up], dim=1)


def make_overlay_base(le, resize, crop):
    le = le.astype(np.float32)
    le = le - le.min()
    le = le / (le.max() + 1e-6)
    le = (le * 255).astype(np.uint8)
    le = cv2.resize(le, (resize, resize), interpolation=cv2.INTER_AREA)
    if crop is not None:
        y0 = (resize - crop) // 2
        x0 = (resize - crop) // 2
        le = le[y0:y0 + crop, x0:x0 + crop]
    return le


def save_overlay(base_gray, heat, out_path, alpha=ALPHA):
    hm = heat.astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-6)
    hm_u8 = (hm * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(base_bgr, 1.0 - alpha, hm_color, alpha, 0)
    cv2.imwrite(out_path, out)


def image_score_from_heat(heat):
    if SCORE_MODE == "max":
        return float(np.max(heat))
    elif SCORE_MODE == "topk":
        flat = heat.flatten()
        k = max(1, int(TOPK_FRAC * flat.size))
        return float(np.mean(np.partition(flat, -k)[-k:]))
    else:
        raise ValueError("SCORE_MODE must be 'max' or 'topk'")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "heatmaps"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "overlays"), exist_ok=True)

    with open(os.path.join(MODEL_DIR, "stats.json"), "r") as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    resize = int(stats["resize"])
    crop = int(stats["crop"])
    eps = float(stats["eps"])
    use_contrast = bool(stats.get("use_contrast", False))

    knn = joblib.load(os.path.join(MODEL_DIR, "knn.joblib"))
    backbone = ResNet18Features().to(DEVICE).eval()

    rows = []

    for split in ["normal", "anomaly"]:
        split_dir = os.path.join(DATA_ROOT, "test", split)
        if not os.path.isdir(split_dir):
            continue
        files = list_npy(split_dir)
        if not files:
            continue

        for p in tqdm(files, desc=f"Testing {split}"):
            le, he = load_le_he_npy(p)
            x = preprocess(le, he, mean, std, eps, resize, crop, use_contrast)
            xt = torch.from_numpy(x).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                f2, f3 = backbone(xt)
                emb = build_patch_embeddings(f2, f3)  # (1,C,h,w)
                _, C, h, w = emb.shape
                patches = emb.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy().astype(np.float32)

            dists, _ = knn.kneighbors(patches, n_neighbors=1, return_distance=True)
            dists = dists.reshape(h, w).astype(np.float32)

            heat = cv2.resize(dists, (crop, crop), interpolation=cv2.INTER_LINEAR)
            score = image_score_from_heat(heat)

            name = os.path.splitext(os.path.basename(p))[0]
            np.save(os.path.join(OUT_DIR, "heatmaps", f"{name}.npy"), heat)

            base = make_overlay_base(le, resize, crop)
            out_overlay = os.path.join(OUT_DIR, "overlays", f"{name}.png")
            save_overlay(base, heat, out_overlay)

            rows.append({"name": name, "split": split, "score": score, "overlay": out_overlay})

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)
    print("Saved:", os.path.join(OUT_DIR, "scores.csv"))


if __name__ == "__main__":
    main()

