"""
LIGHTWEIGHT Hyperparameter Tuning for Parts C & D with Progress Tracking
Fixes: data loading issues, no multiprocessing hangs, streaming output
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from part_a_multimodal_embeddings import (
    BaselineClassifier,
    ImageEmbedder,
    MMIMDbLikeDataset,
    TextEmbedder,
    collate_fn,
    resolve_default_data_dir,
    run_epoch,
    set_seed,
)

PUNCT_RE = re.compile(r"[^\w\s]")


# ============================================================================
# PROGRESS & FORMATTING UTILITIES
# ============================================================================

def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds / 3600
        mins = (seconds % 3600) / 60
        return f"{int(hours)}h{int(mins)}m" if mins > 0 else f"{int(hours)}h"


def _print_header(text: str):
    print(f"\n{'='*90}\n{text:^90}\n{'='*90}\n")


def _print_trial_update(trial_num: int, total_trials: int, acc: float, best_acc: float, elapsed: float):
    """Print formatted trial update."""
    if total_trials == 0:
        return
    pct = (trial_num / total_trials) * 100
    filled = int(40 * trial_num / total_trials)
    bar = "█" * filled + "░" * (40 - filled)
    avg_time = elapsed / max(trial_num, 1)
    remaining = avg_time * (total_trials - trial_num)
    eta_str = _format_time(remaining) if trial_num < total_trials else "Done!"
    
    print(f"  [{bar}] {pct:5.1f}% | Trial {trial_num:3d}/{total_trials:3d} | "
          f"Acc: {acc:.4f} | Best: {best_acc:.4f} | ETA: {eta_str:>8}")


# ============================================================================
# DATA & CONFIG CLASSES
# ============================================================================

@dataclass(frozen=True)
class PartCRunConfig:
    labeled_ratio: float
    pseudo_threshold: float
    consistency_weight: float
    strong_drop_prob: float
    batch_size: int
    lr: float
    embed_dim: int
    seed: int


@dataclass(frozen=True)
class PartDRunConfig:
    embed_dim: int
    batch_size: int
    pretrain_lr: float
    finetune_lr: float
    temperature: float
    seed: int


@dataclass
class RunResult:
    best_val_acc: float
    last_val_acc: float


# ============================================================================
# PART C: SEMI-SUPERVISED LEARNING
# ============================================================================

def split_labeled_unlabeled(train_ds, labeled_ratio: float, seed: int):
    n = len(train_ds)
    if n <= 1:
        return Subset(train_ds, list(range(n))), Subset(train_ds, [])
    labeled_n = max(1, int(labeled_ratio * n))
    labeled_n = min(labeled_n, n - 1)
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    labeled_idx = idx[:labeled_n]
    unlabeled_idx = idx[labeled_n:]
    return Subset(train_ds, labeled_idx), Subset(train_ds, unlabeled_idx)


def weak_text_aug(texts: list[str]) -> list[str]:
    """Light text cleaning for pseudo-labeling."""
    out = []
    for t in texts:
        s = str(t).lower()
        s = PUNCT_RE.sub("", s)
        s = " ".join(s.split())
        out.append(s)
    return out


def strong_text_aug(texts: list[str], drop_prob: float = 0.15) -> list[str]:
    """Strong augmentation for consistency training."""
    out = []
    for text in texts:
        words = str(text).split()
        if len(words) <= 1:
            out.append(str(text))
            continue
        kept = [w for w in words if random.random() > drop_prob]
        if not kept:
            kept = [words[0]]
        if len(kept) > 1 and random.random() < 0.1:
            i = random.randrange(len(kept) - 1)
            kept[i], kept[i + 1] = kept[i + 1], kept[i]
        out.append(" ".join(kept))
    return out


def pseudo_label_batch(logits: torch.Tensor, threshold: float):
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    keep = conf >= threshold
    return pred, keep


def train_semisupervised_epoch(
    labeled_loader, unlabeled_loader, img_model, txt_model, clf,
    optimizer, device, threshold: float, consistency_weight: float, strong_drop_prob: float,
):
    img_model.train()
    txt_model.train()
    clf.train()

    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_steps = 0
    total_kept = 0
    total_unlabeled = 0

    unlabeled_iter = iter(unlabeled_loader)
    for batch_l in labeled_loader:
        try:
            batch_u = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            batch_u = next(unlabeled_iter)

        images_l = batch_l["images"].to(device)
        texts_l = batch_l["texts"]
        labels_l = batch_l["labels"].to(device)

        images_u = batch_u["images"].to(device)
        texts_u = batch_u["texts"]

        logits_l = clf(img_model(images_l), txt_model(texts_l, device))
        loss_sup = ce(logits_l, labels_l)

        texts_u_w = weak_text_aug(texts_u)
        logits_u_w = clf(img_model(images_u), txt_model(texts_u_w, device))
        pseudo, keep = pseudo_label_batch(logits_u_w.detach(), threshold)

        texts_u_s = strong_text_aug(texts_u, drop_prob=strong_drop_prob)
        logits_u_s = clf(img_model(images_u), txt_model(texts_u_s, device))

        if keep.any():
            loss_unsup = ce(logits_u_s[keep], pseudo[keep])
        else:
            loss_unsup = torch.tensor(0.0, device=device)

        loss = loss_sup + consistency_weight * loss_unsup

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        total_kept += int(keep.sum().item())
        total_unlabeled += int(keep.numel())

    avg_loss = total_loss / max(total_steps, 1)
    kept_fraction = total_kept / max(total_unlabeled, 1)
    return avg_loss, kept_fraction


def evaluate(loader, img_model, txt_model, clf, device):
    img_model.eval()
    txt_model.eval()
    clf.eval()
    total_correct = 0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)
            logits = clf(img_model(images), txt_model(texts, device))
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += labels.size(0)

    return total_correct / max(total_items, 1)


def run_part_c_trial(
    ds, cfg: PartCRunConfig, device: torch.device, epochs: int
) -> RunResult:
    """Run one Part C trial (single seed, single config)."""
    set_seed(cfg.seed)
    
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.seed))
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, cfg.labeled_ratio, cfg.seed)

    # NO num_workers to avoid hanging
    labeled_loader = DataLoader(labeled_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    img_model = ImageEmbedder(cfg.embed_dim, backbone="baseline").to(device)
    txt_model = TextEmbedder(cfg.embed_dim, backbone="baseline").to(device)
    clf = BaselineClassifier(cfg.embed_dim, ds.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(clf.parameters()),
        lr=cfg.lr,
    )

    best_val_acc = -1.0
    last_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        _ = train_semisupervised_epoch(
            labeled_loader, unlabeled_loader, img_model, txt_model, clf,
            optimizer, device, cfg.pseudo_threshold, cfg.consistency_weight, cfg.strong_drop_prob,
        )
        val_acc = evaluate(val_loader, img_model, txt_model, clf, device)
        best_val_acc = max(best_val_acc, val_acc)
        last_val_acc = val_acc

    return RunResult(best_val_acc=best_val_acc, last_val_acc=last_val_acc)


# ============================================================================
# PART D: SELF-SUPERVISED LEARNING
# ============================================================================

def info_nce(image_z: torch.Tensor, text_z: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = image_z @ text_z.T / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_i2t + loss_t2i)


def pretrain_epoch(loader, img_model, txt_model, optimizer, device, temperature: float):
    img_model.train()
    txt_model.train()
    total = 0.0
    steps = 0

    for batch in loader:
        images = batch["images"].to(device)
        texts = batch["texts"]

        img_z = img_model(images)
        txt_z = txt_model(texts, device)

        loss = info_nce(img_z, txt_z, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


def run_part_d_trial(
    ds, cfg: PartDRunConfig, device: torch.device,
    pretrain_epochs: int, finetune_epochs: int
) -> RunResult:
    """Run one Part D trial (single seed, single config)."""
    set_seed(cfg.seed)
    
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    img_model = ImageEmbedder(cfg.embed_dim, backbone="baseline").to(device)
    txt_model = TextEmbedder(cfg.embed_dim, backbone="baseline").to(device)

    # Pre-training
    pretrain_optimizer = torch.optim.Adam(
        list(img_model.parameters()) + list(txt_model.parameters()),
        lr=cfg.pretrain_lr,
    )

    for _ in range(pretrain_epochs):
        _ = pretrain_epoch(train_loader, img_model, txt_model, pretrain_optimizer, device, cfg.temperature)

    # Fine-tuning
    clf = BaselineClassifier(cfg.embed_dim, ds.num_classes).to(device)
    finetune_optimizer = torch.optim.Adam(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(clf.parameters()),
        lr=cfg.finetune_lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    last_val_acc = 0.0
    for _ in range(finetune_epochs):
        _ = run_epoch(train_loader, img_model, txt_model, clf, criterion, finetune_optimizer, device, True)
        _, va_acc = run_epoch(val_loader, img_model, txt_model, clf, criterion, finetune_optimizer, device, False)
        best_val_acc = max(best_val_acc, va_acc)
        last_val_acc = va_acc

    return RunResult(best_val_acc=best_val_acc, last_val_acc=last_val_acc)


# ============================================================================
# GRID SEARCH & PARSING
# ============================================================================

def _parse_list(raw: str, cast) -> list:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    return [cast(v) for v in vals]


def _cartesian_grid(space: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    combos = []
    for row in itertools.product(*values):
        combos.append({k: v for k, v in zip(keys, row)})
    return combos


def _sample_trials(grid: List[Dict[str, Any]], max_trials: int | None, seed: int) -> List[Dict[str, Any]]:
    if max_trials is None or max_trials <= 0 or len(grid) <= max_trials:
        return grid
    rng = random.Random(seed)
    idx = list(range(len(grid)))
    rng.shuffle(idx)
    return [grid[i] for i in idx[:max_trials]]


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight GPU Tuning for Parts C & D")
    
    parser.add_argument("--data-dir", type=Path, default=resolve_default_data_dir())
    parser.add_argument("--max-samples", type=int, default=256, help="Limit dataset size")
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--run-part-c", action="store_true", default=True)
    parser.add_argument("--run-part-d", action="store_true", default=False)
    
    # Part C
    parser.add_argument("--part-c-epochs", type=int, default=2)
    parser.add_argument("--part-c-embed-dims", type=str, default="64,128,256")
    parser.add_argument("--part-c-batch-sizes", type=str, default="32,64")
    parser.add_argument("--part-c-lrs", type=str, default="0.001,0.0003")
    parser.add_argument("--part-c-labeled-ratios", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--part-c-pseudo-thresholds", type=str, default="0.85,0.90,0.95")
    parser.add_argument("--part-c-consistency-weights", type=str, default="0.3,0.5,0.8")
    parser.add_argument("--part-c-strong-drop-probs", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--max-trials-part-c", type=int, default=1, help="0 = full grid")
    
    # Part D
    parser.add_argument("--part-d-pretrain-epochs", type=int, default=2)
    parser.add_argument("--part-d-finetune-epochs", type=int, default=2)
    parser.add_argument("--part-d-embed-dims", type=str, default="64,128,256")
    parser.add_argument("--part-d-batch-sizes", type=str, default="32,64")
    parser.add_argument("--part-d-pretrain-lrs", type=str, default="0.001,0.0003")
    parser.add_argument("--part-d-finetune-lrs", type=str, default="0.001,0.0003")
    parser.add_argument("--part-d-temperatures", type=str, default="0.05,0.07,0.10")
    parser.add_argument("--max-trials-part-d", type=int, default=1, help="0 = full grid")
    
    parser.add_argument("--output-dir", type=Path, default=Path("tuning_runs"))
    
    args = parser.parse_args()
    
    # Parse all hyperparameters
    args.part_c_embed_dims = _parse_list(args.part_c_embed_dims, int)
    args.part_c_batch_sizes = _parse_list(args.part_c_batch_sizes, int)
    args.part_c_lrs = _parse_list(args.part_c_lrs, float)
    args.part_c_labeled_ratios = _parse_list(args.part_c_labeled_ratios, float)
    args.part_c_pseudo_thresholds = _parse_list(args.part_c_pseudo_thresholds, float)
    args.part_c_consistency_weights = _parse_list(args.part_c_consistency_weights, float)
    args.part_c_strong_drop_probs = _parse_list(args.part_c_strong_drop_probs, float)
    
    args.part_d_embed_dims = _parse_list(args.part_d_embed_dims, int)
    args.part_d_batch_sizes = _parse_list(args.part_d_batch_sizes, int)
    args.part_d_pretrain_lrs = _parse_list(args.part_d_pretrain_lrs, float)
    args.part_d_finetune_lrs = _parse_list(args.part_d_finetune_lrs, float)
    args.part_d_temperatures = _parse_list(args.part_d_temperatures, float)
    
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        try:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
        except Exception:
            gpu_name = "Unknown GPU"
        device = torch.device(f"cuda:{gpu_idx}")
        print(f"Using GPU {gpu_idx}: {gpu_name} (device={device})")
    else:
        device = torch.device("cpu")
        print("CUDA not available — using CPU")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    _print_header(f"🚀 LIGHTWEIGHT TUNING - Parts C & D | Device: {device}")
    print(f"Output: {out_dir}\n")
    
    ds = MMIMDbLikeDataset(args.data_dir, max_samples=args.max_samples, backbone="baseline")
    
    # ===== PART C =====
    if args.run_part_c:
        _print_header("PART C: SEMI-SUPERVISED LEARNING")
        
        space_c = {
            "embed_dim": args.part_c_embed_dims,
            "batch_size": args.part_c_batch_sizes,
            "lr": args.part_c_lrs,
            "labeled_ratio": args.part_c_labeled_ratios,
            "pseudo_threshold": args.part_c_pseudo_thresholds,
            "consistency_weight": args.part_c_consistency_weights,
            "strong_drop_prob": args.part_c_strong_drop_probs,
        }
        grid_c = _cartesian_grid(space_c)
        trials_c = _sample_trials(grid_c, args.max_trials_part_c, args.seed)
        
        print(f"Grid size: {len(grid_c)} | Sampled: {len(trials_c)} trials\n")
        
        csv_path = out_dir / "part_c_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "embed_dim", "batch_size", "lr", "labeled_ratio",
                "pseudo_threshold", "consistency_weight", "strong_drop_prob",
                "best_val_acc", "last_val_acc"
            ])
            
            best_trial_acc = -1.0
            part_c_start = time.time()
            
            for i, trial_dict in enumerate(trials_c, start=1):
                cfg = PartCRunConfig(
                    labeled_ratio=trial_dict["labeled_ratio"],
                    pseudo_threshold=trial_dict["pseudo_threshold"],
                    consistency_weight=trial_dict["consistency_weight"],
                    strong_drop_prob=trial_dict["strong_drop_prob"],
                    batch_size=trial_dict["batch_size"],
                    lr=trial_dict["lr"],
                    embed_dim=trial_dict["embed_dim"],
                    seed=args.seed,
                )
                
                result = run_part_c_trial(ds, cfg, device, args.part_c_epochs)
                best_trial_acc = max(best_trial_acc, result.best_val_acc)
                
                writer.writerow([
                    i, cfg.embed_dim, cfg.batch_size, cfg.lr, cfg.labeled_ratio,
                    cfg.pseudo_threshold, cfg.consistency_weight, cfg.strong_drop_prob,
                    f"{result.best_val_acc:.4f}", f"{result.last_val_acc:.4f}"
                ])
                f.flush()
                
                elapsed = time.time() - part_c_start
                _print_trial_update(i, len(trials_c), result.best_val_acc, best_trial_acc, elapsed)
        
        part_c_time = time.time() - part_c_start
        print(f"\n✓ Part C complete | Time: {_format_time(part_c_time)} | Best: {best_trial_acc:.4f}")
        print(f"  Results: {csv_path}\n")
    
    # ===== PART D =====
    if args.run_part_d:
        _print_header("PART D: SELF-SUPERVISED LEARNING")
        
        space_d = {
            "embed_dim": args.part_d_embed_dims,
            "batch_size": args.part_d_batch_sizes,
            "pretrain_lr": args.part_d_pretrain_lrs,
            "finetune_lr": args.part_d_finetune_lrs,
            "temperature": args.part_d_temperatures,
        }
        grid_d = _cartesian_grid(space_d)
        trials_d = _sample_trials(grid_d, args.max_trials_part_d, args.seed)
        
        print(f"Grid size: {len(grid_d)} | Sampled: {len(trials_d)} trials")
        print(f"Per trial: {args.part_d_pretrain_epochs} pretrain + {args.part_d_finetune_epochs} finetune epochs\n")
        
        csv_path = out_dir / "part_d_results.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "trial", "embed_dim", "batch_size", "pretrain_lr", "finetune_lr",
                "temperature", "best_val_acc", "last_val_acc"
            ])
            
            best_trial_acc = -1.0
            part_d_start = time.time()
            
            for i, trial_dict in enumerate(trials_d, start=1):
                cfg = PartDRunConfig(
                    embed_dim=trial_dict["embed_dim"],
                    batch_size=trial_dict["batch_size"],
                    pretrain_lr=trial_dict["pretrain_lr"],
                    finetune_lr=trial_dict["finetune_lr"],
                    temperature=trial_dict["temperature"],
                    seed=args.seed,
                )
                
                result = run_part_d_trial(ds, cfg, device, args.part_d_pretrain_epochs, args.part_d_finetune_epochs)
                best_trial_acc = max(best_trial_acc, result.best_val_acc)
                
                writer.writerow([
                    i, cfg.embed_dim, cfg.batch_size, cfg.pretrain_lr, cfg.finetune_lr,
                    cfg.temperature, f"{result.best_val_acc:.4f}", f"{result.last_val_acc:.4f}"
                ])
                f.flush()
                
                elapsed = time.time() - part_d_start
                _print_trial_update(i, len(trials_d), result.best_val_acc, best_trial_acc, elapsed)
        
        part_d_time = time.time() - part_d_start
        print(f"\n✓ Part D complete | Time: {_format_time(part_d_time)} | Best: {best_trial_acc:.4f}")
        print(f"  Results: {csv_path}\n")
    
    _print_header(f"✅ ALL DONE! | Output: {out_dir}")


if __name__ == "__main__":
    main()
