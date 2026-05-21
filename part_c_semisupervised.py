"""Option 7 Part C: Semi-supervised learning with config comparison.

This script runs pseudo-labeling plus consistency training for one or more
configurations, then ranks the runs by validation accuracy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List
import copy
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
    set_seed,
)


@dataclass(frozen=True)
class PartCRunConfig:
    backbone: str
    labeled_ratio: float
    pseudo_threshold: float
    consistency_weight: float
    strong_drop_prob: float
    seed: int


@dataclass
class PartCRunResult:
    config: PartCRunConfig
    best_val_acc: float
    best_epoch: int
    supervised_best_val_acc: float
    last_val_acc: float
    last_val_loss: float
    last_val_macro_f1: float
    last_train_loss: float
    last_kept_fraction: float
    checkpoint_path: str
    metrics_path: str


PUNCT_RE = re.compile(r"[^\w\s]")


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
    """Light, meaning-preserving text cleaning used for pseudo-labeling."""
    out = []
    for t in texts:
        s = str(t).lower()
        s = PUNCT_RE.sub("", s)
        s = " ".join(s.split())
        out.append(s)
    return out


def strong_text_aug(texts: list[str], drop_prob: float = 0.15) -> list[str]:
    """Stronger augmentation used for the consistency target."""
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


from sklearn.metrics import precision_recall_fscore_support

def _macro_metrics_from_labels(preds: torch.Tensor, trues: torch.Tensor) -> dict:
    if preds.numel() == 0 or trues.numel() == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    p, r, f1, _ = precision_recall_fscore_support(
        trues.numpy(), preds.numpy(), average="macro", zero_division=0
    )
    return {"macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}


def train_semisupervised_epoch(
    labeled_loader,
    unlabeled_loader,
    img_model,
    txt_model,
    clf,
    optimizer,
    device,
    threshold: float,
    consistency_weight: float,
    strong_drop_prob: float,
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
    return avg_loss, total_kept, total_unlabeled, kept_fraction


def evaluate(loader, img_model, txt_model, clf, device):
    img_model.eval()
    txt_model.eval()
    clf.eval()
    total_correct = 0
    total_items = 0
    total_loss = 0.0
    preds = []
    trues = []
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)
            logits = clf(img_model(images), txt_model(texts, device))
            total_loss += ce(logits, labels).item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += labels.size(0)
            preds.append(logits.argmax(1).cpu())
            trues.append(labels.cpu())

    if total_items == 0:
        return {"loss": 0.0, "acc": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    preds = torch.cat(preds)
    trues = torch.cat(trues)
    macro_metrics = _macro_metrics_from_labels(preds, trues)
    return {
        "loss": total_loss / total_items,
        "acc": total_correct / total_items,
        "macro_precision": macro_metrics["macro_precision"],
        "macro_recall": macro_metrics["macro_recall"],
        "macro_f1": macro_metrics["macro_f1"],
    }


def _parse_list(raw: str, cast) -> list:
    if raw is None:
        return []
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return [cast(value) for value in values]


def _build_config_grid(args) -> List[PartCRunConfig]:
    backbones = args.backbones if args.compare_configs and args.backbones else [args.backbone]
    labeled_ratios = _parse_list(args.labeled_ratios, float) if args.compare_configs else [args.labeled_ratio]
    pseudo_thresholds = _parse_list(args.pseudo_thresholds, float) if args.compare_configs else [args.pseudo_threshold]
    consistency_weights = _parse_list(args.consistency_weights, float) if args.compare_configs else [args.consistency_weight]
    strong_drop_probs = _parse_list(args.strong_drop_probs, float) if args.compare_configs and args.strong_drop_probs else [args.strong_drop_prob]
    seeds = _parse_list(args.seeds, int) if args.compare_configs and args.seeds else [args.seed]

    configs: List[PartCRunConfig] = []
    for backbone, labeled_ratio, pseudo_threshold, consistency_weight, strong_drop_prob, seed in itertools.product(
        backbones,
        labeled_ratios,
        pseudo_thresholds,
        consistency_weights,
        strong_drop_probs,
        seeds,
    ):
        configs.append(
            PartCRunConfig(
                backbone=backbone,
                labeled_ratio=labeled_ratio,
                pseudo_threshold=pseudo_threshold,
                consistency_weight=consistency_weight,
                strong_drop_prob=strong_drop_prob,
                seed=seed,
            )
        )
    return configs


def _config_id(cfg: PartCRunConfig) -> str:
    return (
        f"backbone-{cfg.backbone}_lab-{cfg.labeled_ratio:.2f}_pt-{cfg.pseudo_threshold:.2f}_"
        f"cw-{cfg.consistency_weight:.2f}_sd-{cfg.strong_drop_prob:.2f}_seed-{cfg.seed}"
    )


def _train_one_config(args, cfg: PartCRunConfig, run_dir: Path, device: torch.device) -> PartCRunResult:
    set_seed(cfg.seed)
    ds = MMIMDbLikeDataset(args.data_dir, max_samples=args.max_samples, backbone=cfg.backbone)
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.seed))
    labeled_ds, unlabeled_ds = split_labeled_unlabeled(train_ds, cfg.labeled_ratio, cfg.seed)

    labeled_loader = DataLoader(labeled_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    img_model = ImageEmbedder(args.embed_dim, backbone=cfg.backbone).to(device)
    txt_model = TextEmbedder(args.embed_dim, backbone=cfg.backbone).to(device)
    clf = BaselineClassifier(args.embed_dim, ds.num_classes).to(device)

    optimizer = torch.optim.Adam(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(clf.parameters()),
        lr=args.lr,
    )


    # --- SUPERVISED BASELINE ---
    print(f"Running supervised baseline on labeled data only (ratio={cfg.labeled_ratio})")
    img_model_sup = copy.deepcopy(img_model)
    txt_model_sup = copy.deepcopy(txt_model)
    clf_sup = copy.deepcopy(clf)
    optimizer_sup = torch.optim.Adam(
        list(img_model_sup.parameters()) + list(txt_model_sup.parameters()) + list(clf_sup.parameters()),
        lr=args.lr,
    )
    sup_criterion = nn.CrossEntropyLoss()
    supervised_best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        img_model_sup.train()
        txt_model_sup.train()
        clf_sup.train()
        for batch in labeled_loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)
            logits = clf_sup(img_model_sup(images), txt_model_sup(texts, device))
            loss = sup_criterion(logits, labels)
            optimizer_sup.zero_grad()
            loss.backward()
            optimizer_sup.step()
        
        val_stats = evaluate(val_loader, img_model_sup, txt_model_sup, clf_sup, device)
        supervised_best_val_acc = max(supervised_best_val_acc, float(val_stats.get("acc", 0.0)))
    print(f"Supervised baseline best val_acc: {supervised_best_val_acc:.4f}")
    # ---------------------------

    best_val_acc = -1.0
    best_epoch = 0
    last_val_acc = 0.0
    last_train_loss = 0.0
    last_kept_fraction = 0.0
    last_val_loss = 0.0
    last_val_macro_f1 = 0.0

    cfg_id = _config_id(cfg)
    checkpoint_path = run_dir / f"{cfg_id}_best.pt"
    metrics_path = run_dir / f"{cfg_id}_metrics.jsonl" if args.save_metrics else None

    print(f"Using dataset dir: {args.data_dir}")
    print(f"Backbone: {cfg.backbone}")
    print(f"Config: {cfg_id}")
    print(f"Train labeled ratio: {cfg.labeled_ratio}")
    print(f"Pseudo threshold: {cfg.pseudo_threshold}")
    print(f"Consistency weight: {cfg.consistency_weight}")
    print(f"Strong drop prob: {cfg.strong_drop_prob}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, kept_count, unlabeled_count, kept_fraction = train_semisupervised_epoch(
            labeled_loader,
            unlabeled_loader,
            img_model,
            txt_model,
            clf,
            optimizer,
            device,
            cfg.pseudo_threshold,
            cfg.consistency_weight,
            cfg.strong_drop_prob,
        )
        val_stats = evaluate(val_loader, img_model, txt_model, clf, device)
        val_acc = float(val_stats.get("acc", 0.0))
        val_loss = float(val_stats.get("loss", 0.0))
        val_macro_f1 = float(val_stats.get("macro_f1", 0.0))
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            if args.save_best_model:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "config": cfg.__dict__,
                        "epoch": epoch,
                        "best_val_acc": best_val_acc,
                        "img_model": img_model.state_dict(),
                        "txt_model": txt_model.state_dict(),
                        "clf": clf.state_dict(),
                    },
                    checkpoint_path,
                )

        last_val_acc = val_acc
        last_train_loss = tr_loss
        last_kept_fraction = kept_fraction
        last_val_loss = val_loss
        last_val_macro_f1 = val_macro_f1

        if metrics_path is not None:
            row = {
                "run_id": cfg_id,
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_macro_f1": val_macro_f1,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "kept_count": kept_count,
                "unlabeled_count": unlabeled_count,
                "kept_fraction": kept_fraction,
                "improved": improved,
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

        marker = "*" if improved else " "
        print(
            f"{marker} Epoch {epoch:>3}/{args.epochs} | train_loss={tr_loss:.4f} | val_acc={val_acc:.4f} | "
            f"best={best_val_acc:.4f} @ {best_epoch} | kept={kept_count}/{unlabeled_count} ({kept_fraction:.2%})"
        )

    return PartCRunResult(
        config=cfg,
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        supervised_best_val_acc=supervised_best_val_acc,
        last_val_acc=last_val_acc,
        last_val_loss=last_val_loss,
        last_val_macro_f1=last_val_macro_f1,
        last_train_loss=last_train_loss,
        last_kept_fraction=last_kept_fraction,
        checkpoint_path=str(checkpoint_path) if args.save_best_model else "",
        metrics_path=str(metrics_path) if metrics_path else "",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=resolve_default_data_dir())
    parser.add_argument("--backbone", choices=["baseline", "clip"], default="baseline")
    parser.add_argument("--backbones", nargs="+", choices=["baseline", "clip"], default=None, help="Optional list of backbones to compare")
    parser.add_argument("--compare-configs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--labeled-ratio", type=float, default=0.2)
    parser.add_argument("--labeled-ratios", type=str, default="0.1,0.2,0.5")
    parser.add_argument("--pseudo-threshold", type=float, default=0.9)
    parser.add_argument("--pseudo-thresholds", type=str, default="0.85,0.9,0.95")
    parser.add_argument("--consistency-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weights", type=str, default="0.3,0.5,0.8")
    parser.add_argument("--strong-drop-prob", type=float, default=0.15, help="Word dropout prob for strong augmentation")
    parser.add_argument("--strong-drop-probs", type=str, default="0.1,0.15,0.2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=str, default=None, help="Optional comma-separated seeds for repeated runs")
    parser.add_argument("--output-dir", type=Path, default=Path("part_c_runs"))
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=False, help="Save per-config metrics.jsonl files when enabled.")
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report-top-k", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.output_dir / datetime.now().strftime("part_c_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
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

    configs = _build_config_grid(args)
    print(f"Using device: {device}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {run_dir}")
    print(f"Compare configs: {args.compare_configs}")
    print(f"Total configs: {len(configs)}")
    print(f"Batch size: {args.batch_size} | Epochs: {args.epochs} | lr: {args.lr}")

    results: List[PartCRunResult] = []
    for run_index, cfg in enumerate(configs, start=1):
        print(f"\n=== Run {run_index}/{len(configs)} ===")
        results.append(_train_one_config(args, cfg, run_dir, device))

    results = sorted(results, key=lambda r: (r.best_val_acc, r.last_val_acc), reverse=True)
    summary_csv = run_dir / "summary.csv"
    summary_json = run_dir / "summary.json"
    report_md = run_dir / "report.md"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "run_id",
                "backbone",
                "labeled_ratio",
                "pseudo_threshold",
                "consistency_weight",
                "strong_drop_prob",
                "seed",
                "lr",
                "batch_size",
                "best_val_acc",
                "supervised_best_val_acc",
                "best_epoch",
                "last_val_acc",
                "last_val_loss",
                "last_val_macro_f1",
                "last_train_loss",
                "last_kept_fraction",
                "checkpoint_path",
                "metrics_path",
            ]
        )
        for result in results:
            cfg = result.config
            writer.writerow(
                [
                    _config_id(cfg),
                    cfg.backbone,
                    f"{cfg.labeled_ratio:.4f}",
                    f"{cfg.pseudo_threshold:.4f}",
                    f"{cfg.consistency_weight:.4f}",
                    f"{cfg.strong_drop_prob:.4f}",
                    cfg.seed,
                    args.lr,
                    args.batch_size,
                    f"{result.best_val_acc:.6f}",
                    f"{result.supervised_best_val_acc:.6f}",
                    result.best_epoch,
                    f"{result.last_val_acc:.6f}",
                    f"{result.last_val_loss:.6f}",
                    f"{result.last_val_macro_f1:.6f}",
                    f"{result.last_train_loss:.6f}",
                    f"{result.last_kept_fraction:.6f}",
                    result.checkpoint_path,
                    result.metrics_path,
                ]
            )

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "run_id": _config_id(result.config),
                    "backbone": result.config.backbone,
                    "labeled_ratio": result.config.labeled_ratio,
                    "pseudo_threshold": result.config.pseudo_threshold,
                    "consistency_weight": result.config.consistency_weight,
                    "strong_drop_prob": result.config.strong_drop_prob,
                    "seed": result.config.seed,
                    "best_val_acc": result.best_val_acc,
                    "supervised_best_val_acc": result.supervised_best_val_acc,
                    "best_epoch": result.best_epoch,
                    "last_val_acc": result.last_val_acc,
                    "last_val_loss": result.last_val_loss,
                    "last_val_macro_f1": result.last_val_macro_f1,
                    "last_train_loss": result.last_train_loss,
                    "last_kept_fraction": result.last_kept_fraction,
                    "checkpoint_path": result.checkpoint_path,
                    "metrics_path": result.metrics_path,
                }
                for result in results
            ],
            f,
            indent=2,
        )

    best = results[0] if results else None
    print("\n=== Best Run ===")
    if best is not None:
        cfg = best.config
        print(
            f"best_val_acc={best.best_val_acc:.4f} at epoch {best.best_epoch} | backbone={cfg.backbone} | "
            f"labeled_ratio={cfg.labeled_ratio} | pseudo_threshold={cfg.pseudo_threshold} | "
            f"consistency_weight={cfg.consistency_weight} | strong_drop_prob={cfg.strong_drop_prob} | seed={cfg.seed}"
        )
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")
    if args.save_best_model and best is not None:
        print(f"Best checkpoint: {best.checkpoint_path}")


if __name__ == "__main__":
    main()
