"""
Option 7 Part D: Self-supervised Learning
Contrastive image-text pretraining + supervised fine-tuning starter.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import csv
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from part_a_multimodal_embeddings import (
    BaselineClassifier,
    ImageEmbedder,
    MMIMDbLikeDataset,
    TextEmbedder,
    collate_fn,
    evaluate_loader,
    resolve_default_data_dir,
    run_epoch,
    set_seed,
)


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


def representation_similarity(loader, img_model, txt_model, device):
    img_model.eval()
    txt_model.eval()
    match_sum = 0.0
    random_sum = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            img_z = img_model(images)
            txt_z = txt_model(texts, device)
            match_sum += float((img_z * txt_z).sum(dim=1).sum().item())
            random_txt = txt_z[torch.randperm(txt_z.size(0), device=txt_z.device)]
            random_sum += float((img_z * random_txt).sum(dim=1).sum().item())
            total += int(images.size(0))

    return {
        "match_cosine": match_sum / max(total, 1),
        "random_cosine": random_sum / max(total, 1),
        "similarity_gap": (match_sum - random_sum) / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=resolve_default_data_dir())
    parser.add_argument("--backbone", choices=["baseline", "clip"], default="baseline")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory to write run outputs (metrics, checkpoints).")
    parser.add_argument("--save-checkpoint", action="store_true", help="Save best finetune checkpoint.")
    parser.add_argument("--save-metrics", action="store_true", help="Save per-epoch metrics to metrics.jsonl.")
    parser.add_argument("--compare-scratch", action="store_true", help="Also run a finetune-only baseline from scratch for transfer comparison.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
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
    print(f"Using dataset dir: {args.data_dir}")
    print(f"Using backbone: {args.backbone}")

    ds = MMIMDbLikeDataset(args.data_dir, max_samples=args.max_samples, backbone=args.backbone)
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    # Use num_workers=0 to avoid multiprocessing deadlocks in some environments
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    img_model = ImageEmbedder(args.embed_dim, backbone=args.backbone).to(device)
    txt_model = TextEmbedder(args.embed_dim, backbone=args.backbone).to(device)

    pretrain_optimizer = torch.optim.Adam(
        list(img_model.parameters()) + list(txt_model.parameters()),
        lr=args.lr,
    )

    # Prepare output directory
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("part_d_runs") / f"run_{ts}"
    else:
        out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.jsonl"
    best_ckpt_path = out_dir / "part_d_best.pt"

    best_val = -1.0
    best_epoch = None
    pretrain_similarity = {"match_cosine": 0.0, "random_cosine": 0.0, "similarity_gap": 0.0}

    for epoch in range(1, args.pretrain_epochs + 1):
        pre_loss = pretrain_epoch(
            train_loader,
            img_model,
            txt_model,
            pretrain_optimizer,
            device,
            args.temperature,
        )
        print(f"Pretrain {epoch}/{args.pretrain_epochs} | contrastive loss={pre_loss:.4f}")
        if args.save_metrics:
            with open(metrics_path, "a", encoding="utf8") as f:
                json.dump({
                    "phase": "pretrain",
                    "epoch": epoch,
                    "pretrain_loss": float(pre_loss),
                }, f)
                f.write("\n")

    pretrain_similarity = representation_similarity(val_loader, img_model, txt_model, device)
    print(
        f"Pretrain representation similarity | match={pretrain_similarity['match_cosine']:.4f} | "
        f"random={pretrain_similarity['random_cosine']:.4f} | gap={pretrain_similarity['similarity_gap']:.4f}"
    )
    if args.save_metrics:
        with open(metrics_path, "a", encoding="utf8") as f:
            json.dump({"phase": "representation", "epoch": 0, **pretrain_similarity}, f)
            f.write("\n")

    # Fine-tuning stage
    clf = BaselineClassifier(args.embed_dim, ds.num_classes).to(device)
    finetune_optimizer = torch.optim.Adam(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(clf.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_scratch_val = None
    if args.compare_scratch:
        scratch_img = ImageEmbedder(args.embed_dim, backbone=args.backbone).to(device)
        scratch_txt = TextEmbedder(args.embed_dim, backbone=args.backbone).to(device)
        scratch_clf = BaselineClassifier(args.embed_dim, ds.num_classes).to(device)
        scratch_opt = torch.optim.Adam(
            list(scratch_img.parameters()) + list(scratch_txt.parameters()) + list(scratch_clf.parameters()),
            lr=args.lr,
        )
        scratch_best = -1.0
        for epoch in range(1, args.finetune_epochs + 1):
            run_epoch(train_loader, scratch_img, scratch_txt, scratch_clf, criterion, scratch_opt, device, True)
            _, scratch_val_acc = run_epoch(val_loader, scratch_img, scratch_txt, scratch_clf, criterion, device, False)
            scratch_best = max(scratch_best, scratch_val_acc)
        best_scratch_val = scratch_best
        print(f"Scratch baseline best val acc={best_scratch_val:.4f}")

    for epoch in range(1, args.finetune_epochs + 1):
        tr_loss, tr_acc = run_epoch(
            train_loader,
            img_model,
            txt_model,
            clf,
            criterion,
            finetune_optimizer,
            device,
            True,
        )
        va_loss, va_acc = run_epoch(
            val_loader,
            img_model,
            txt_model,
            clf,
            criterion,
            finetune_optimizer,
            device,
            False,
        )
        val_stats = evaluate_loader(val_loader, img_model, txt_model, clf, criterion, device)
        va_loss = float(val_stats.get("loss", va_loss))
        va_acc = float(val_stats.get("acc", va_acc))
        va_macro_f1 = float(val_stats.get("macro_f1", 0.0))
        print(
            f"Finetune {epoch}/{args.finetune_epochs} "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_macro_f1:.4f}"
        )
        if args.save_metrics:
            with open(metrics_path, "a", encoding="utf8") as f:
                json.dump({
                    "phase": "finetune",
                    "epoch": epoch,
                    "train_loss": float(tr_loss),
                    "train_acc": float(tr_acc),
                    "val_loss": float(va_loss),
                    "val_acc": float(va_acc),
                    "val_macro_f1": float(va_macro_f1),
                }, f)
                f.write("\n")

        # Save best checkpoint by validation accuracy
        if va_acc > best_val:
            best_val = float(va_acc)
            best_epoch = epoch
            if args.save_checkpoint:
                best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "img_state_dict": img_model.state_dict(),
                        "txt_state_dict": txt_model.state_dict(),
                        "clf_state_dict": clf.state_dict(),
                        "epoch": epoch,
                        "best_val": best_val,
                    },
                    best_ckpt_path,
                )

    # Write summary
    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "run_time",
            "max_samples",
            "embed_dim",
            "batch_size",
            "pretrain_epochs",
            "finetune_epochs",
            "best_val",
            "best_epoch",
            "pretrain_match_cosine",
            "pretrain_random_cosine",
            "pretrain_similarity_gap",
            "scratch_best_val",
            "transfer_delta",
        ])
        writer.writerow([
            datetime.now().isoformat(),
            args.max_samples,
            args.embed_dim,
            args.batch_size,
            args.pretrain_epochs,
            args.finetune_epochs,
            best_val,
            best_epoch,
            pretrain_similarity["match_cosine"],
            pretrain_similarity["random_cosine"],
            pretrain_similarity["similarity_gap"],
            best_scratch_val if best_scratch_val is not None else "",
            (best_val - best_scratch_val) if best_scratch_val is not None else "",
        ])

    print(f"Run outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
