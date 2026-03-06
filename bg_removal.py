import os, json, math, random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, DFineForObjectDetection


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainCfg:
    checkpoint: str = "ustc-community/dfine_n_coco"
    images_dir: str = "data/multiview_npy"     # <-- your 3ch preprocessed .npy folder
    ann_json: str = "data/annotations.json"   # COCO format JSON

    class_names: Tuple[str, ...] = ("bone",)  # single-class
    epochs: int = 30
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.1
    num_workers: int = 4

    amp: bool = True
    grad_accum: int = 1
    freeze_backbone_epochs: int = 1

    out_dir: str = "runs/dfine_nano_multiview"
    seed: int = 0


CFG = TrainCfg()
EPS = 1e-6


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_hwc3(x: np.ndarray) -> np.ndarray:
    """Accept (H,W,3) or (3,H,W), return (H,W,3)."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {x.shape}")
    if x.shape[-1] == 3:
        return x
    if x.shape[0] == 3:
        return np.transpose(x, (1, 2, 0))
    raise ValueError(f"Cannot infer channel axis for shape {x.shape}")


def mv01_to_uint8(mv01: np.ndarray) -> np.ndarray:
    """
    Your multiview input is float32 in [0,1].
    Convert to uint8 RGB-like array for HF processor.
    """
    mv01 = mv01.astype(np.float32, copy=False)
    mv01 = np.clip(mv01, 0.0, 1.0)
    return (mv01 * 255.0 + 0.5).astype(np.uint8)


def load_coco(coco_json_path: str):
    with open(coco_json_path, "r") as f:
        coco = json.load(f)
    images_by_id = {img["id"]: img for img in coco["images"]}
    ann_by_image = {img_id: [] for img_id in images_by_id.keys()}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        if img_id in ann_by_image:
            ann_by_image[img_id].append(ann)
    return coco, images_by_id, ann_by_image


# ----------------------------
# Dataset
# ----------------------------
class MultiViewCocoDataset(Dataset):
    """
    Loads 3-channel multiview npy (float32 [0,1]) and COCO bboxes.
    Uses AutoImageProcessor to create DETR-style labels.
    """
    def __init__(
        self,
        images_dir: str,
        coco_json: str,
        processor: AutoImageProcessor,
        category_id_to_label: Dict[int, int],
    ):
        self.images_dir = images_dir
        self.processor = processor
        self.category_id_to_label = category_id_to_label

        _, self.images_by_id, self.ann_by_image = load_coco(coco_json)
        self.image_ids = sorted(self.images_by_id.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        img_info = self.images_by_id[image_id]

        # IMPORTANT: file_name must point to your preprocessed multiview .npy
        npy_path = os.path.join(self.images_dir, img_info["file_name"])
        mv = np.load(npy_path)  # (H,W,3) float32 [0,1] (or (3,H,W))
        mv = ensure_hwc3(mv)

        # Convert to uint8 for processor
        img_u8 = mv01_to_uint8(mv)

        anns = []
        for a in self.ann_by_image.get(image_id, []):
            cat_id = a["category_id"]
            if cat_id not in self.category_id_to_label:
                continue
            bbox = a["bbox"]  # [x,y,w,h] in pixels
            anns.append({
                "bbox": bbox,
                "category_id": self.category_id_to_label[cat_id],
                "area": a.get("area", float(bbox[2] * bbox[3])),
                "iscrowd": 0,
            })

        target = {"image_id": image_id, "annotations": anns}

        encoding = self.processor(images=img_u8, annotations=target, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in encoding.items()}


def make_collate_fn(processor: AutoImageProcessor):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = [b["pixel_values"] for b in batch]
        padded = processor.pad({"pixel_values": pixel_values}, return_tensors="pt")
        out = {"pixel_values": padded["pixel_values"]}
        if "pixel_mask" in padded:
            out["pixel_mask"] = padded["pixel_mask"]
        out["labels"] = [b["labels"] for b in batch]
        return out
    return collate_fn


# ----------------------------
# Train
# ----------------------------
def main(cfg: TrainCfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Processor: keep defaults; we already give uint8 images.
    processor = AutoImageProcessor.from_pretrained(cfg.checkpoint)

    # Model: change head to your classes
    model = DFineForObjectDetection.from_pretrained(
        cfg.checkpoint,
        num_labels=len(cfg.class_names),
        ignore_mismatched_sizes=True,
    )

    model.config.id2label = {i: n for i, n in enumerate(cfg.class_names)}
    model.config.label2id = {n: i for i, n in enumerate(cfg.class_names)}

    # Build category_id -> contiguous label map from COCO categories
    coco, _, _ = load_coco(cfg.ann_json)
    name_to_catid = {c["name"]: c["id"] for c in coco.get("categories", [])}
    category_id_to_label = {}
    for i, name in enumerate(cfg.class_names):
        if name not in name_to_catid:
            raise ValueError(f"Class '{name}' not found in JSON categories. Found: {list(name_to_catid.keys())}")
        category_id_to_label[name_to_catid[name]] = i

    train_ds = MultiViewCocoDataset(cfg.images_dir, cfg.ann_json, processor, category_id_to_label)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=make_collate_fn(processor),
    )

    model.to(device)

    # Optional speed optimization (PyTorch 2.x)
    if hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    model.train()
    optim.zero_grad(set_to_none=True)

    for epoch in range(cfg.epochs):
        # warmup freeze backbone (helps small datasets)
        if epoch < cfg.freeze_backbone_epochs:
            model.model.backbone.requires_grad_(False)
        else:
            model.model.backbone.requires_grad_(True)

        running = 0.0
        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch.get("pixel_mask", None)
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(device, non_blocking=True)

            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                out = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
                loss = out.loss / cfg.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            running += loss.item() * cfg.grad_accum

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"epoch {epoch+1:03d}/{cfg.epochs} step {step+1:04d}/{len(train_loader)} loss {running/(step+1):.4f}")

        # Save
        save_dir = os.path.join(cfg.out_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(save_dir, exist_ok=True)
        model_to_save = model._orig_mod if hasattr(model, "_orig_mod") else model
        model_to_save.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"[OK] saved {save_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main(CFG)


import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional

EPS = 1e-6

@dataclass
class MultiViewCfg:
    # Robust scaling (calibration-free)
    raw_clip_p: Tuple[float, float] = (0.5, 99.5)
    norm8_clip_p: Tuple[float, float] = (0.5, 99.5)

    # Very light denoise
    denoise_raw: bool = True
    denoise_ksize: int = 3  # median 3x3 is safe

    # Edge view (safe, deterministic)
    edge_blur_ksize: int = 3       # small pre-blur before Sobel
    edge_strength: float = 0.75    # how much edge magnitude is added to raw view
    edge_p: Tuple[float, float] = (5.0, 99.0)  # robust normalize gradient magnitude


def robust01(x: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
    """Robust normalize to [0,1] using percentiles (per image)."""
    x = x.astype(np.float32, copy=False)
    lo = np.percentile(x, p_low)
    hi = np.percentile(x, p_high)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + EPS)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def median_denoise01(x01: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median blur on [0,1] image via uint8."""
    u8 = (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)
    u8 = cv2.medianBlur(u8, ksize)
    return u8.astype(np.float32) / 255.0


def edge_enhanced_view_from_raw(raw01: np.ndarray, cfg: MultiViewCfg) -> np.ndarray:
    """
    Create an edge-enhanced view from raw01 using Sobel magnitude.
    Returns [0,1] float32.
    """
    u8 = (np.clip(raw01, 0.0, 1.0) * 255.0).astype(np.uint8)

    if cfg.edge_blur_ksize and cfg.edge_blur_ksize > 1:
        u8 = cv2.GaussianBlur(u8, (cfg.edge_blur_ksize, cfg.edge_blur_ksize), 0)

    gx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    lo = np.percentile(mag, cfg.edge_p[0])
    hi = np.percentile(mag, cfg.edge_p[1])
    mag01 = (mag - lo) / (hi - lo + EPS)
    mag01 = np.clip(mag01, 0.0, 1.0).astype(np.float32)

    # Edge-enhanced (not pure edges): helps small bones without destroying tone
    out = raw01 + cfg.edge_strength * mag01
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def ensure_gray(x: np.ndarray) -> np.ndarray:
    """Ensure 2D grayscale float/uint input returns 2D array."""
    if x.ndim == 2:
        return x
    if x.ndim == 3 and x.shape[-1] == 1:
        return x[..., 0]
    raise ValueError(f"Expected grayscale 2D, got shape {x.shape}")


def rgb8_to_gray01(rgb8: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB/BGR to grayscale [0,1]."""
    if rgb8.ndim != 3 or rgb8.shape[-1] != 3:
        raise ValueError(f"Expected (H,W,3) RGB8, got {rgb8.shape}")
    # If your array is BGR (OpenCV), use cv2.COLOR_BGR2GRAY instead.
    gray = cv2.cvtColor(rgb8, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def make_multiview_input(
    raw16: np.ndarray,
    gray8_norm: np.ndarray,
    rgb8: Optional[np.ndarray] = None,
    cfg: MultiViewCfg = MultiViewCfg(),
) -> np.ndarray:
    """
    Build a 3-channel input (H,W,3) float32 in [0,1]:
      ch0 = raw16 robust normalized
      ch1 = gray8_norm stabilized
      ch2 = edge-enhanced view from raw16

    rgb8 is optional; included only if you want to sanity-check or swap views later.
    """
    raw16 = ensure_gray(raw16)
    gray8_norm = ensure_gray(gray8_norm)

    # View A: raw16 -> [0,1]
    raw01 = robust01(raw16, *cfg.raw_clip_p)

    # Optional light denoise on raw view only
    if cfg.denoise_raw:
        raw01 = median_denoise01(raw01, cfg.denoise_ksize)

    # View B: 8-bit normalized gray -> [0,1] (stabilize with robust01 in case it isn't perfect)
    # If gray8_norm is already uint8 0..255, robust01 handles it fine.
    g01 = robust01(gray8_norm, *cfg.norm8_clip_p)

    # View C: edge-enhanced from raw
    edge01 = edge_enhanced_view_from_raw(raw01, cfg)

    # Stack into multiview (H,W,3)
    mv = np.stack([raw01, g01, edge01], axis=-1).astype(np.float32)

    return mv


# ----------------------------
# Example usage with .npy
# ----------------------------
if __name__ == "__main__":
    raw16 = np.load("raw16.npy")          # (H,W) uint16 or float
    gray8 = np.load("gray8_norm.npy")     # (H,W) uint8 or float
    rgb8  = np.load("rgb8.npy")           # (H,W,3) uint8 (optional)

    mv = make_multiview_input(raw16, gray8, rgb8=rgb8)
    np.save("multiview_input.npy", mv)    # (H,W,3) float32 in [0,1]


from sklearn.decomposition import PCA

PCA_DIM = 256  # try 128, 256, 512 depending on speed/accuracy

pca = PCA(n_components=PCA_DIM, whiten=True, random_state=0)
memory_p = pca.fit_transform(memory)

# save PCA + projected memory
import joblib
joblib.dump(pca, os.path.join(OUT_DIR, "pca.joblib"))
np.save(os.path.join(OUT_DIR, "memory_pca.npy"), memory_p)

# fit knn on projected memory
knn = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean")
knn.fit(memory_p)
joblib.dump(knn, os.path.join(OUT_DIR, "knn.joblib"))


pca = joblib.load(os.path.join(MODEL_DIR, "pca.joblib"))

patches_p = pca.transform(patches)  # patches is (num_patches, C)
dists, _ = knn.kneighbors(patches_p, n_neighbors=5, return_distance=True)
patch_scores = dists.mean(axis=1).astype(np.float32)

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

