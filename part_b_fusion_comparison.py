"""
Option 7 Part B: Early vs Late Fusion Comparison
Compare baseline encoders against CLIP with the same fusion heads.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime

from part_a_multimodal_embeddings import (
    ImageEmbedder,
    MMIMDbLikeDataset,
    TextEmbedder,
    collate_fn,
    resolve_default_data_dir,
    set_seed,
)

# Early fusion
class EarlyFusionClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([img_emb, txt_emb], dim=1)
        return self.head(fused)


class EarlyFusionAdvancedClassifier(nn.Module):
    """Advanced early fusion with projections, interactions, residual MLP and aux heads."""

    def __init__(self, embed_dim: int, num_classes: int, dropout_p: float = 0.2, modality_dropout_p: float = 0.1):
        super().__init__()
        self.modality_dropout_p = modality_dropout_p

        self.img_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))
        self.txt_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.LayerNorm(embed_dim))

        hidden = 256
        in_dim = embed_dim * 4
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden, num_classes)

        self.img_aux_head = nn.Linear(embed_dim, num_classes)
        self.txt_aux_head = nn.Linear(embed_dim, num_classes)

    def _apply_modality_dropout(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_p <= 0:
            return img, txt

        bs = img.size(0)
        drop_img = (torch.rand(bs, 1, device=img.device) < self.modality_dropout_p).float()
        drop_txt = (torch.rand(bs, 1, device=txt.device) < self.modality_dropout_p).float()

        # Keep at least one modality present for each sample.
        both = (drop_img > 0) & (drop_txt > 0)
        if both.any():
            drop_txt = drop_txt.masked_fill(both, 0.0)

        return img * (1.0 - drop_img), txt * (1.0 - drop_txt)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        img_n = F.normalize(img_emb, dim=1)
        txt_n = F.normalize(txt_emb, dim=1)

        img_p = self.img_proj(img_n)
        txt_p = self.txt_proj(txt_n)
        img_p, txt_p = self._apply_modality_dropout(img_p, txt_p)

        inter_mul = img_p * txt_p
        inter_abs = torch.abs(img_p - txt_p)
        fused = torch.cat([img_p, txt_p, inter_mul, inter_abs], dim=1)

        h1 = torch.relu(self.fc1(fused))
        h1 = self.dropout(h1)
        h2 = torch.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h = h1 + h2

        logits = self.out(h)
        img_aux = self.img_aux_head(img_p)
        txt_aux = self.txt_aux_head(txt_p)
        return logits, img_aux, txt_aux

# Late fusion
class LateFusionClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.img_head = nn.Linear(embed_dim, num_classes)
        self.txt_head = nn.Linear(embed_dim, num_classes)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_logits = self.img_head(img_emb)
        txt_logits = self.txt_head(txt_emb)
        return 0.5 * img_logits + 0.5 * txt_logits


class LateFusionAdvancedClassifier(nn.Module):
    """Advanced late fusion with learnable calibration and sample-dependent weighting."""

    def __init__(self, embed_dim: int, num_classes: int, modality_dropout_p: float = 0.1):
        super().__init__()
        self.modality_dropout_p = modality_dropout_p

        self.img_head = nn.Linear(embed_dim, num_classes)
        self.txt_head = nn.Linear(embed_dim, num_classes)

        self.log_temp_img = nn.Parameter(torch.zeros(1))
        self.log_temp_txt = nn.Parameter(torch.zeros(1))

        self.alpha_net = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _apply_modality_dropout(self, img: torch.Tensor, txt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.modality_dropout_p <= 0:
            return img, txt

        bs = img.size(0)
        drop_img = (torch.rand(bs, 1, device=img.device) < self.modality_dropout_p).float()
        drop_txt = (torch.rand(bs, 1, device=txt.device) < self.modality_dropout_p).float()

        both = (drop_img > 0) & (drop_txt > 0)
        if both.any():
            drop_txt = drop_txt.masked_fill(both, 0.0)

        return img * (1.0 - drop_img), txt * (1.0 - drop_txt)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_n = F.normalize(img_emb, dim=1)
        txt_n = F.normalize(txt_emb, dim=1)
        img_n, txt_n = self._apply_modality_dropout(img_n, txt_n)

        t_img = torch.exp(self.log_temp_img).clamp(min=1e-3)
        t_txt = torch.exp(self.log_temp_txt).clamp(min=1e-3)
        img_logits = self.img_head(img_n) / t_img
        txt_logits = self.txt_head(txt_n) / t_txt

        alpha = torch.sigmoid(self.alpha_net(torch.cat([img_n, txt_n], dim=1)))
        return alpha * img_logits + (1.0 - alpha) * txt_logits

# clip
class ClipStyleFusionClassifier(nn.Module):
    """CLIP-style fusion using normalized modality averaging."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_n = F.normalize(img_emb, dim=1)
        txt_n = F.normalize(txt_emb, dim=1)
        fused = F.normalize(0.5 * (img_n + txt_n), dim=1)
        return self.head(fused)

# GNN 
class TwoNodeMessagePassing(nn.Module):
    """Tiny message-passing block for a 2-node graph (image <-> text)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(embed_dim, embed_dim)
        self.neigh_lin = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, img_h: torch.Tensor, txt_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img_new = self.self_lin(img_h) + self.neigh_lin(txt_h)
        txt_new = self.self_lin(txt_h) + self.neigh_lin(img_h)
        img_new = torch.relu(self.norm(img_new))
        txt_new = torch.relu(self.norm(txt_new))
        return img_new, txt_new


class EarlyGNNFusionClassifier(nn.Module):
    """Early fusion through a small 2-node GNN before classification."""

    def __init__(self, embed_dim: int, num_classes: int, num_layers: int = 2):
        super().__init__()
        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.txt_proj = nn.Linear(embed_dim, embed_dim)
        self.layers = nn.ModuleList([TwoNodeMessagePassing(embed_dim) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_h = torch.relu(self.img_proj(F.normalize(img_emb, dim=1)))
        txt_h = torch.relu(self.txt_proj(F.normalize(txt_emb, dim=1)))
        for layer in self.layers:
            img_h, txt_h = layer(img_h, txt_h)
        return self.head(torch.cat([img_h, txt_h], dim=1))


class LateGNNFusionClassifier(nn.Module):
    """Late fusion with separate heads and a GNN-informed dynamic fusion gate."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.img_head = nn.Linear(embed_dim, num_classes)
        self.txt_head = nn.Linear(embed_dim, num_classes)

        self.img_proj = nn.Linear(embed_dim, embed_dim)
        self.txt_proj = nn.Linear(embed_dim, embed_dim)
        self.mp = TwoNodeMessagePassing(embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        img_logits = self.img_head(img_emb)
        txt_logits = self.txt_head(txt_emb)

        img_h = torch.relu(self.img_proj(F.normalize(img_emb, dim=1)))
        txt_h = torch.relu(self.txt_proj(F.normalize(txt_emb, dim=1)))
        img_h, txt_h = self.mp(img_h, txt_h)

        alpha = torch.sigmoid(self.gate(torch.cat([img_h, txt_h], dim=1)))
        return alpha * img_logits + (1.0 - alpha) * txt_logits


def run_epoch_fusion(loader, img_model, txt_model, model, criterion, optimizer, device, train: bool, aux_weight: float = 0.2):
    img_model.train(train)
    txt_model.train(train)
    model.train(train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            img_emb = img_model(images)
            txt_emb = txt_model(texts, device)

            out = model(img_emb, txt_emb)
            if isinstance(out, tuple):
                logits, img_aux, txt_aux = out
                loss = (
                    criterion(logits, labels)
                    + aux_weight * criterion(img_aux, labels)
                    + aux_weight * criterion(txt_aux, labels)
                )
            else:
                logits = out
                loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += labels.size(0)

    return total_loss / max(total_items, 1), total_correct / max(total_items, 1)


from sklearn.metrics import precision_recall_fscore_support

def _macro_metrics_from_labels(preds: torch.Tensor, trues: torch.Tensor) -> dict:
    if preds.numel() == 0 or trues.numel() == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    p, r, f1, _ = precision_recall_fscore_support(
        trues.numpy(), preds.numpy(), average="macro", zero_division=0
    )
    return {"macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}


def evaluate_fusion(loader, img_model, txt_model, model, criterion, device, aux_weight: float = 0.2):
    img_model.eval()
    txt_model.eval()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0
    combined_preds = []
    image_only_preds = []
    text_only_preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            img_emb = img_model(images)
            txt_emb = txt_model(texts, device)

            out = model(img_emb, txt_emb)
            if isinstance(out, tuple):
                logits, img_aux, txt_aux = out
                loss = (
                    criterion(logits, labels)
                    + aux_weight * criterion(img_aux, labels)
                    + aux_weight * criterion(txt_aux, labels)
                )
            else:
                logits = out
                loss = criterion(logits, labels)

            zero_txt = torch.zeros_like(txt_emb)
            zero_img = torch.zeros_like(img_emb)
            img_only_out = model(img_emb, zero_txt)
            txt_only_out = model(zero_img, txt_emb)
            if isinstance(img_only_out, tuple):
                img_only_logits = img_only_out[0]
                txt_only_logits = txt_only_out[0]
            else:
                img_only_logits = img_only_out
                txt_only_logits = txt_only_out

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += labels.size(0)
            combined_preds.append(logits.argmax(1).cpu())
            image_only_preds.append(img_only_logits.argmax(1).cpu())
            text_only_preds.append(txt_only_logits.argmax(1).cpu())
            trues.append(labels.cpu())

    if total_items == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "image_only_acc": 0.0,
            "text_only_acc": 0.0,
            "combined_acc": 0.0,
            "fusion_gain": 0.0,
        }

    combined_preds = torch.cat(combined_preds)
    image_only_preds = torch.cat(image_only_preds)
    text_only_preds = torch.cat(text_only_preds)
    trues = torch.cat(trues)
    combined_metrics = _macro_metrics_from_labels(combined_preds, trues)
    image_only_acc = float((image_only_preds == trues).float().mean().item())
    text_only_acc = float((text_only_preds == trues).float().mean().item())
    combined_acc = float((combined_preds == trues).float().mean().item())

    return {
        "loss": total_loss / total_items,
        "acc": total_correct / total_items,
        "macro_precision": combined_metrics["macro_precision"],
        "macro_recall": combined_metrics["macro_recall"],
        "macro_f1": combined_metrics["macro_f1"],
        "image_only_acc": image_only_acc,
        "text_only_acc": text_only_acc,
        "combined_acc": combined_acc,
        "fusion_gain": combined_acc - max(image_only_acc, text_only_acc),
    }


def train_one_fusion(
    backbone: str,
    fusion: str,
    ds,
    train_loader,
    val_loader,
    embed_dim,
    epochs,
    lr,
    device,
    img_model,
    txt_model,
    init_img_state,
    init_txt_state,
    adv_dropout,
    modality_dropout,
    weight_decay,
    label_smoothing,
    aux_weight,
    gnn_layers: int = 2,
    optimizer_name: str = "adam",
    save_best: bool = True,
    save_metrics: bool = False,
    out_dir: Path | None = None,
):
    
    # Reset to initial pretrained state so early/late runs are comparable.
    img_model.load_state_dict(init_img_state)
    txt_model.load_state_dict(init_txt_state)

    if fusion == "early":
        model = EarlyFusionClassifier(embed_dim, ds.num_classes).to(device)
    elif fusion == "early_adv":
        model = EarlyFusionAdvancedClassifier(
            embed_dim,
            ds.num_classes,
            dropout_p=adv_dropout,
            modality_dropout_p=modality_dropout,
        ).to(device)
    elif fusion == "late":
        model = LateFusionClassifier(embed_dim, ds.num_classes).to(device)
    elif fusion == "late_adv":
        model = LateFusionAdvancedClassifier(
            embed_dim,
            ds.num_classes,
            modality_dropout_p=modality_dropout,
        ).to(device)
    elif fusion == "clip":
        model = ClipStyleFusionClassifier(embed_dim, ds.num_classes).to(device)
    elif fusion == "early_gnn":
        model = EarlyGNNFusionClassifier(embed_dim, ds.num_classes, num_layers=gnn_layers).to(device)
    elif fusion == "late_gnn":
        model = LateGNNFusionClassifier(embed_dim, ds.num_classes).to(device)
    else:
        raise ValueError(
            "fusion must be one of: 'early', 'early_adv', 'late', 'late_adv', 'clip', 'early_gnn', 'late_gnn'"
        )

    opt_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # prepare per-run output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if out_dir is None:
        out_root = Path("part_b_runs") / f"part_b_{ts}" / f"{backbone}_{fusion}"
    else:
        out_root = Path(out_dir) / f"{backbone}_{fusion}"
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "metrics.jsonl" if save_metrics else None
    checkpoint_path = out_root / f"backbone-{backbone}_fusion-{fusion}_best.pt"

    best_val = 0.0
    best_epoch = 0
    val_accs: list[float] = []
    val_f1s: list[float] = []
    fusion_gains: list[float] = []
    image_only_accs: list[float] = []
    text_only_accs: list[float] = []
    metrics_handle = open(metrics_path, "w", encoding="utf-8") if metrics_path else None
    try:
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch_fusion(
                train_loader,
                img_model,
                txt_model,
                model,
                criterion,
                optimizer,
                device,
                True,
                aux_weight=aux_weight,
            )
            val_stats = evaluate_fusion(val_loader, img_model, txt_model, model, criterion, device, aux_weight=aux_weight)
            va_loss = float(val_stats.get("loss", 0.0))
            va_acc = float(val_stats.get("acc", 0.0))
            val_accs.append(va_acc)
            val_f1s.append(float(val_stats.get("macro_f1", 0.0)))
            fusion_gains.append(float(val_stats.get("fusion_gain", 0.0)))
            image_only_accs.append(float(val_stats.get("image_only_acc", 0.0)))
            text_only_accs.append(float(val_stats.get("text_only_acc", 0.0)))
            print(
                f"[{backbone}/{fusion}] epoch {epoch}/{epochs} "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}"
            )
            rec = {
                "epoch": epoch,
                "train_loss": float(tr_loss),
                "train_acc": float(tr_acc),
                "val_loss": float(va_loss),
                "val_acc": float(va_acc),
                "macro_precision": float(val_stats.get("macro_precision", 0.0)),
                "macro_recall": float(val_stats.get("macro_recall", 0.0)),
                "macro_f1": float(val_stats.get("macro_f1", 0.0)),
                "image_only_acc": float(val_stats.get("image_only_acc", 0.0)),
                "text_only_acc": float(val_stats.get("text_only_acc", 0.0)),
                "combined_acc": float(val_stats.get("combined_acc", va_acc)),
                "fusion_gain": float(val_stats.get("fusion_gain", 0.0)),
            }
            if metrics_handle is not None:
                metrics_handle.write(json.dumps(rec) + "\n")

            if va_acc > best_val:
                best_val = va_acc
                best_epoch = epoch
                if save_best:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"img": img_model.state_dict(), "txt": txt_model.state_dict(), "model": model.state_dict()}, checkpoint_path)
    finally:
        if metrics_handle is not None:
            metrics_handle.close()

    return {
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "final_val_acc": float(val_accs[-1]) if val_accs else 0.0,
        "mean_val_acc": float(sum(val_accs) / len(val_accs)) if val_accs else 0.0,
        "final_macro_f1": float(val_f1s[-1]) if val_f1s else 0.0,
        "mean_macro_f1": float(sum(val_f1s) / len(val_f1s)) if val_f1s else 0.0,
        "final_fusion_gain": float(fusion_gains[-1]) if fusion_gains else 0.0,
        "mean_fusion_gain": float(sum(fusion_gains) / len(fusion_gains)) if fusion_gains else 0.0,
        "final_image_only_acc": float(image_only_accs[-1]) if image_only_accs else 0.0,
        "final_text_only_acc": float(text_only_accs[-1]) if text_only_accs else 0.0,
        "metrics": str(metrics_path) if metrics_path else None,
        "checkpoint": str(checkpoint_path) if (best_val > 0 and save_best) else None,
        "config_used": {
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "label_smoothing": float(label_smoothing),
            "aux_weight": float(aux_weight),
            "modality_dropout": float(modality_dropout),
            "adv_dropout": float(adv_dropout),
            "gnn_layers": int(gnn_layers),
            "batch_size": int(getattr(train_loader, "batch_size", 0) or 0),
        },
    }


def run_backbone(backbone: str, args) -> dict[str, dict]:
    import numpy as np
    import copy
    from torch.utils.data import DataLoader, random_split
    ds = MMIMDbLikeDataset(args.data_dir, max_samples=args.max_samples, backbone=backbone)

    def _load_latest_tuning(path_root: Path) -> dict | None:
        root = Path(path_root)
        if not root.exists():
            return None
        subs = sorted([p for p in root.iterdir() if p.is_dir()])
        if not subs:
            return None
        latest = subs[-1]
        cfg_file = latest / "best_configs.json"
        if not cfg_file.exists():
            return None
        try:
            return json.loads(cfg_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    tuning = _load_latest_tuning(Path(__file__).resolve().parent / "tuning_runs") if getattr(args, 'use_tuning_best', False) else None
    if tuning and tuning.get("part_b") and tuning["part_b"].get(backbone):
        first_fusion = next(iter(tuning["part_b"][backbone].keys()))
        bs = tuning["part_b"][backbone][first_fusion].get("best_config", {}).get("batch_size", args.batch_size)
        args.batch_size = bs

    print(f"\n=== Backbone: {backbone} ({args.runs} runs per fusion head) ===")
    
    # Store aggregated metrics
    scores_agg: dict[str, dict] = {}
    metadata: dict[str, dict] = {}

    for fusion in args.fusion_modes:
        print(f"\n--- Fusion Head: {fusion} ---")
        run_results = []
        for run_idx in range(args.runs):
            run_seed = args.seed + run_idx
            set_seed(run_seed)

            train_n = int(0.8 * len(ds))
            val_n = len(ds) - train_n
            train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(run_seed))

            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

            img_model_run = ImageEmbedder(args.embed_dim, backbone=backbone).to(args.device)
            txt_model_run = TextEmbedder(args.embed_dim, backbone=backbone).to(args.device)
            init_img_state = copy.deepcopy(img_model_run.state_dict())
            init_txt_state = copy.deepcopy(txt_model_run.state_dict())

            lr_local = args.lr
            adv_dropout_local = args.adv_dropout
            modality_dropout_local = args.modality_dropout
            label_smoothing_local = args.label_smoothing
            aux_weight_local = args.aux_weight
            gnn_layers_local = 2

            if tuning and tuning.get("part_b") and tuning["part_b"].get(backbone):
                fusion_entry = tuning["part_b"][backbone].get(fusion, {})
                cfg = fusion_entry.get("best_config", {})
                if cfg:
                    lr_local = cfg.get("lr", lr_local)
                    args.optimizer = cfg.get("optimizer", args.optimizer)
                    adv_dropout_local = cfg.get("adv_dropout", adv_dropout_local)
                    modality_dropout_local = cfg.get("modality_dropout", modality_dropout_local)
                    label_smoothing_local = cfg.get("label_smoothing", label_smoothing_local)
                    aux_weight_local = cfg.get("aux_weight", aux_weight_local)
                gnn_layers_local = cfg.get("gnn_layers", gnn_layers_local)

            res = train_one_fusion(
                backbone,
                fusion,
                ds,
                train_loader,
                val_loader,
                args.embed_dim,
                args.epochs,
                lr_local,
                args.device,
                img_model_run,
                txt_model_run,
                init_img_state,
                init_txt_state,
                adv_dropout_local,
                modality_dropout_local,
                args.weight_decay,
                label_smoothing_local,
                aux_weight_local,
                gnn_layers_local,
                args.optimizer,
                save_best=args.save_best_model,
                save_metrics=args.save_run_metrics,
                out_dir=args.output_dir,
            )
            print(f"  Run {run_idx+1}/{args.runs} Best Val: {res['best_val']:.4f}")
            run_results.append(res)
            
        vals = [r['best_val'] for r in run_results]
        f1s = [r['final_macro_f1'] for r in run_results]
        gains = [r['final_fusion_gain'] for r in run_results]

        agg_res = {
            "best_val_mean": float(np.mean(vals)),
            "best_val_std": float(np.std(vals)),
            "final_macro_f1_mean": float(np.mean(f1s)),
            "final_macro_f1_std": float(np.std(f1s)),
            "final_fusion_gain_mean": float(np.mean(gains)),
            "final_fusion_gain_std": float(np.std(gains)),
        }
        
        scores_agg[fusion] = agg_res

    return scores_agg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=resolve_default_data_dir())
    parser.add_argument("--backbone", choices=["baseline", "clip"], default="baseline")
    parser.add_argument("--compare-backbones", action=argparse.BooleanOptionalAction, default=True, 
    help="Compare both baseline and clip backbones (default: true). Use --no-compare-backbones for single-backbone mode.",
    )
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument(
        "--fusion-modes",
        nargs="+",
        choices=["early", "early_adv", "late", "late_adv", "clip", "early_gnn", "late_gnn"],
        default=["early", "late", "clip", "early_gnn", "late_gnn"],
        help="Fusion heads to run and compare.",
    )
    parser.add_argument(
        "--advanced-comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run a full comparison suite (early/early_adv/late/late_adv/clip/early_gnn/late_gnn).",
    )
    parser.add_argument("--aux-weight", type=float, default=0.2, help="Auxiliary loss weight for early_adv.")
    parser.add_argument("--modality-dropout", type=float, default=0.1, help="Drop probability for each modality in advanced methods.")
    parser.add_argument("--adv-dropout", type=float, default=0.2, help="Dropout for advanced early fusion head.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing for CE loss.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1, help="Number of random seeds per configuration.")
    parser.add_argument("--use-tuning-best", action=argparse.BooleanOptionalAction, default=True,
                        help="If true, load best hyperparameters from tuning_runs/<latest>/best_configs.json when available")
    parser.add_argument("--clf-hidden-dim", type=int, default=128)
    parser.add_argument("--clf-dropout", type=float, default=0.2)
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-run-metrics", action=argparse.BooleanOptionalAction, default=False, help="Save per-epoch metrics.jsonl for each fusion run.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Root directory for per-run outputs")
    args = parser.parse_args()

    if args.advanced_comparison:
        args.fusion_modes = ["early", "early_adv", "late", "late_adv", "clip", "early_gnn", "late_gnn"]

    set_seed(args.seed)
    if torch.cuda.is_available():
        gpu_idx = torch.cuda.current_device()
        try:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
        except Exception:
            gpu_name = "Unknown GPU"
        args.device = torch.device(f"cuda:{gpu_idx}")
        print(f"Using GPU {gpu_idx}: {gpu_name} (device={args.device})")
    else:
        args.device = torch.device("cpu")
        print("CUDA not available — using CPU")
    print(f"Using dataset dir: {args.data_dir}")
    print(f"Using backbone: {args.backbone}")
    print(f"Compare backbones: {args.compare_backbones}")
    print(f"Fusion modes: {args.fusion_modes}")

    results: dict[str, dict[str, float]] = {}
    backbones = ["baseline", "clip"] if args.compare_backbones else [args.backbone]
    run_rows = []
    for backbone in backbones:
        backbone_scores = run_backbone(backbone, args)
        results[backbone] = backbone_scores
        for fusion, score in backbone_scores.items():
            row = {"backbone": backbone, "fusion": fusion}
            row.update(score)
            run_rows.append(row)

    run_rows = sorted(run_rows, key=lambda row: row.get("best_val_acc", row.get("best_val_mean", 0)), reverse=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else Path("part_b_runs") / f"part_b_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    summary_json = out_root / "summary.json"

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "backbone",
            "fusion",
            "best_val_mean",
            "best_val_std",
            "final_macro_f1_mean",
            "final_macro_f1_std",
            "final_fusion_gain_mean",
            "final_fusion_gain_std",
        ])
        for row in run_rows:
            writer.writerow([
                f"{row['backbone']}_{row['fusion']}",
                row['backbone'],
                row['fusion'],
                f"{row.get('best_val_mean', 0):.4f}",
                f"{row.get('best_val_std', 0):.4f}",
                f"{row.get('final_macro_f1_mean', 0):.4f}",
                f"{row.get('final_macro_f1_std', 0):.4f}",
                f"{row.get('final_fusion_gain_mean', 0):.4f}",
                f"{row.get('final_fusion_gain_std', 0):.4f}",
            ])

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(run_rows, f, indent=2)

    print("\n=== Fusion Summary ===")
    for backbone, scores in results.items():
        line = " ".join([f"{k}={v.get('best_val_mean', 0.0):.4f}" for k, v in scores.items()])
        print(f"{backbone}: {line}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
