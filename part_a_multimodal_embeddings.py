"""
Option 7 Part A: Text + Image Embeddings Baseline
Choose between a pretrained baseline and CLIP with one flag.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from datetime import datetime
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel, AutoTokenizer, CLIPModel


BASELINE_TEXT_MODEL = "distilbert-base-uncased"
CLIP_MODEL = "openai/clip-vit-base-patch32"

# helpers functions
def _to_feature_tensor(output) -> torch.Tensor:
    """Safely extracts a 2D feature tensor from various HuggingFace outputs."""
    if torch.is_tensor(output):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        hidden = output.last_hidden_state
        return hidden[:, 0] if hidden.dim() == 3 else hidden
    if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
        first = output[0]
        return first[:, 0] if first.dim() == 3 else first
    raise TypeError(f"Expected tensor-like output, got {type(output)}")


def resolve_default_data_dir() -> Path: 
    """Finds the dataset directory either in sam or parent dir"""
    script_dir = Path(__file__).resolve().parent
    # Check common locations for the dataset
    candidates = [
        script_dir,
        script_dir.parent,
    ]
    for path in candidates:
        if (path / "IMDB_four_genre_larger_plot_description.csv").exists():
            return path
    return script_dir # default fallback

def _pick_column(columns: list[str], choices: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for c in choices:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def build_image_transform(backbone: str):
    if backbone == "clip":
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# dataset and dataloaders

class MMIMDbLikeDataset(Dataset):
    """
    A multimodal dataset for text and images.
    Automatically finds the right columns for ID, text, and labels from a CSV.
    """
    def __init__(self, data_dir: Path, max_samples: int = -1, backbone: str = "baseline"):
        self.data_dir = data_dir
        self.backbone = backbone
        
        self.csv_path = data_dir / "IMDB_four_genre_larger_plot_description.csv"
        self.image_base_dir = data_dir / "IMDB four_genre_posters"
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find IMDB_four_genre_larger_plot_description.csv in: {data_dir}")
        self.is_imdb = True

        df = pd.read_csv(self.csv_path, on_bad_lines="skip")
        id_col = _pick_column(list(df.columns), ["id", "image_id", "movie_id", "imdbid"])
        text_col = _pick_column(
            list(df.columns),
            ["productDisplayName", "plot", "text", "overview", "description", "title", "review", "sentence"],
        )
        label_col = _pick_column(
            list(df.columns),
            ["masterCategory", "genre", "label", "labels", "category", "class", "sentiment"],
        )

        if text_col is None or label_col is None:
            raise ValueError(
                "Could not infer id/text/label columns. "
                f"Found columns: {list(df.columns)}"
            )

        if id_col is None:
            id_col = "__row_id__"
            df[id_col] = np.arange(len(df)).astype(str)

        self.id_col = id_col
        self.text_col = text_col
        self.label_col = label_col

        df = df[[self.id_col, self.text_col, self.label_col]].dropna().reset_index(drop=True)
        if max_samples > 0:
            df = df.sample(min(max_samples, len(df)), random_state=42).reset_index(drop=True)
        self.df = df

        self._build_labels()
        
        # Pre-map image paths to avoid slow searching since IMDB has subdirectories
        self.img_paths = {}
        if self.image_base_dir.exists():
            for p in self.image_base_dir.rglob("*.*"):
                if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.img_paths[p.stem] = p

        self.transform = build_image_transform(self.backbone)

    def _build_labels(self) -> None:
        cats = sorted(self.df[self.label_col].astype(str).unique())
        self.label2id = {c: i for i, c in enumerate(cats)}
        self.num_classes = len(self.label2id)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        pid = str(row[self.id_col]).strip()

        label = self.label2id.get(str(row[self.label_col]), 0)
        text = str(row[self.text_col])

        img_tensor = torch.zeros(3, 224, 224)
        img_path = self.img_paths.get(pid)
        
        # fallback for standard flat folder if not in dict
        if not img_path:
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = self.image_base_dir / f"{pid}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break

        if img_path and img_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = self.transform(img)
            except Exception:
                img_tensor = torch.zeros(3, 224, 224)

        return {"image": img_tensor, "text": text, "label": label}


def collate_fn(batch: List[Dict]) -> Dict:
    """Stacks images and labels into tensors, keeps texts as lists."""
    images = torch.stack([b["image"] for b in batch])
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {"images": images, "texts": texts, "labels": labels}



# Models
class TextEmbedder(nn.Module):
    """Encodes text to a fixed-size embedding using either DistilBERT or CLIP."""
    def __init__(self, embed_dim: int, backbone: str = "baseline", freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim

        if backbone == "baseline":
            self.tokenizer = AutoTokenizer.from_pretrained(BASELINE_TEXT_MODEL)
            self.model = AutoModel.from_pretrained(BASELINE_TEXT_MODEL)
            hidden_dim = self.model.config.hidden_size
        elif backbone == "clip":
            self.tokenizer = AutoTokenizer.from_pretrained(CLIP_MODEL)
            self.model = CLIPModel.from_pretrained(CLIP_MODEL)
            hidden_dim = self.model.config.projection_dim
        else:
            raise ValueError("backbone must be 'baseline' or 'clip'")

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        if self.backbone == "clip":
            max_length = self.model.config.text_config.max_position_embeddings
        else:
            max_length = 128

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            if self.backbone == "baseline":
                outputs = self.model(**encoded)
                pooled = outputs.last_hidden_state[:, 0]
            else:
                pooled = _to_feature_tensor(self.model.get_text_features(**encoded))

        return F.normalize(self.proj(pooled), dim=1)


class ImageEmbedder(nn.Module):
    """Encodes images to a fixed-size embedding using either ResNet18 or CLIP."""
    def __init__(self, embed_dim: int, backbone: str = "baseline", freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone

        if backbone == "baseline":
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            hidden_dim = 512
            if freeze_backbone:
                for param in self.features.parameters():
                    param.requires_grad = False
        elif backbone == "clip":
            self.model = CLIPModel.from_pretrained(CLIP_MODEL)
            hidden_dim = self.model.config.projection_dim
            if freeze_backbone:
                for param in self.model.parameters():
                    param.requires_grad = False
        else:
            raise ValueError("backbone must be 'baseline' or 'clip'")

        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.backbone == "baseline":
            x = self.features(images).view(images.size(0), -1)
        else:
            x = _to_feature_tensor(self.model.get_image_features(pixel_values=images))
        x = self.proj(x)
        return F.normalize(x, dim=1)


class BaselineClassifier(nn.Module):
    """Simple MLP that takes concatenated image and text embeddings to predict classes."""
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 128, dropout_p: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([img_emb, txt_emb], dim=1))



# Training and evaluation

def run_epoch(loader, img_model, txt_model, clf, criterion, optimizer, device, train: bool) -> tuple[float, float]:
    """Runs one full pass over the dataset (training or validation)."""
    img_model.train(train)
    txt_model.train(train)
    clf.train(train)

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
            logits = clf(img_emb, txt_emb)
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
    """Calculates macro metrics using sklearn """
    if preds.numel() == 0 or trues.numel() == 0:
        return {"macro_precision": 0.0, "macro_recall": 0.0, "macro_f1": 0.0}

    p, r, f1, _ = precision_recall_fscore_support(
        trues.numpy(), preds.numpy(), average="macro", zero_division=0
    )
    return {"macro_precision": float(p), "macro_recall": float(r), "macro_f1": float(f1)}


def evaluate_loader(loader, img_model, txt_model, clf, criterion, device) -> dict:
    img_model.eval()
    txt_model.eval()
    clf.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    combined_preds = []
    image_only_preds = []
    text_only_preds = []
    trues = []
    match_cos_sum = 0.0
    random_cos_sum = 0.0
    similarity_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            texts = batch["texts"]
            labels = batch["labels"].to(device)

            img_emb = img_model(images)
            txt_emb = txt_model(texts, device)
            logits = clf(img_emb, txt_emb)
            loss = criterion(logits, labels)

            zero_txt = torch.zeros_like(txt_emb)
            zero_img = torch.zeros_like(img_emb)
            img_only_logits = clf(img_emb, zero_txt)
            txt_only_logits = clf(zero_img, txt_emb)

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_items += labels.size(0)

            combined_preds.append(logits.argmax(1).cpu())
            image_only_preds.append(img_only_logits.argmax(1).cpu())
            text_only_preds.append(txt_only_logits.argmax(1).cpu())
            trues.append(labels.cpu())

            match_cos_sum += float((img_emb * txt_emb).sum(dim=1).sum().item())
            random_txt = txt_emb[torch.randperm(txt_emb.size(0), device=txt_emb.device)]
            random_cos_sum += float((img_emb * random_txt).sum(dim=1).sum().item())
            similarity_count += int(labels.size(0))

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
            "match_cosine": 0.0,
            "random_cosine": 0.0,
            "confusion_matrix": [],
        }

    combined_preds = torch.cat(combined_preds)
    image_only_preds = torch.cat(image_only_preds)
    text_only_preds = torch.cat(text_only_preds)
    trues = torch.cat(trues)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(trues.numpy(), combined_preds.numpy()).tolist()

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
        "match_cosine": match_cos_sum / max(similarity_count, 1),
        "random_cosine": random_cos_sum / max(similarity_count, 1),
        "total": int(total_items),
        "confusion_matrix": cm,
    }

# main run and with reporting 

def _load_latest_tuning(path_root: Path) -> Optional[dict]:
    """Helper to auto-load best hyperparameters from tuning if available."""
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


def _best_part_a_config(args, backbone: str, tuning: Optional[dict]) -> dict:
    cfg = {
        "embed_dim": int(args.embed_dim),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "optimizer": str(args.optimizer),
        "clf_hidden_dim": int(args.clf_hidden_dim),
        "clf_dropout": float(args.clf_dropout),
    }
    if args.use_tuning_best and tuning and tuning.get("part_a") and tuning["part_a"].get(backbone):
        best_cfg = tuning["part_a"][backbone].get("best_config", {})
        if best_cfg:
            print(f"Loaded Part A best config for {backbone} from tuning_runs, applying defaults:", best_cfg)
            cfg["embed_dim"] = int(best_cfg.get("embed_dim", cfg["embed_dim"]))
            cfg["batch_size"] = int(best_cfg.get("batch_size", cfg["batch_size"]))
            cfg["lr"] = float(best_cfg.get("lr", cfg["lr"]))
            cfg["weight_decay"] = float(best_cfg.get("weight_decay", cfg["weight_decay"]))
            cfg["optimizer"] = str(best_cfg.get("optimizer", cfg["optimizer"]))
            cfg["clf_hidden_dim"] = int(best_cfg.get("clf_hidden_dim", cfg["clf_hidden_dim"]))
            cfg["clf_dropout"] = float(best_cfg.get("clf_dropout", cfg["clf_dropout"]))
    return cfg


def _run_part_a_backbone(backbone: str, args, tuning: Optional[dict], run_root: Path, device: torch.device) -> dict:
    set_seed(args.seed)
    cfg = _best_part_a_config(args, backbone, tuning)

    ds = MMIMDbLikeDataset(args.data_dir, args.max_samples, backbone=backbone)
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn)

    img_model = ImageEmbedder(cfg["embed_dim"], backbone=backbone).to(device)
    txt_model = TextEmbedder(cfg["embed_dim"], backbone=backbone).to(device)
    clf = BaselineClassifier(cfg["embed_dim"], ds.num_classes, hidden_dim=cfg["clf_hidden_dim"], dropout_p=cfg["clf_dropout"]).to(device)

    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(clf.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    from pathlib import Path
    out_root = Path(run_root) / backbone if args.compare_backbones else Path(run_root)
    out_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_root / f"backbone-{backbone}_best.pt"
    # Ensure it's fully resolved
    out_root.mkdir(parents=True, exist_ok=True)
    metrics_path = out_root / "metrics.jsonl" if args.save_metrics else None
    summary_csv = out_root / "summary.csv"
    summary_path = out_root / "summary.json"
    checkpoint_path = out_root / f"backbone-{backbone}_best.pt"

    best_val = 0.0
    best_epoch = 0
    val_accs: list[float] = []
    val_f1s: list[float] = []
    val_image_only_accs: list[float] = []
    val_text_only_accs: list[float] = []
    val_match_cosines: list[float] = []
    val_random_cosines: list[float] = []
    metrics_handle = open(metrics_path, "w", encoding="utf-8") if metrics_path else None
    try:
        for epoch in range(1, args.epochs + 1):
            tr_loss, tr_acc = run_epoch(train_loader, img_model, txt_model, clf, criterion, optimizer, device, True)
            val_stats = evaluate_loader(val_loader, img_model, txt_model, clf, criterion, device)
            va_loss = val_stats.get("loss", 0.0)
            va_acc = val_stats.get("acc", 0.0)
            val_accs.append(va_acc)
            val_f1s.append(float(val_stats.get("macro_f1", 0.0)))
            val_image_only_accs.append(float(val_stats.get("image_only_acc", 0.0)))
            val_text_only_accs.append(float(val_stats.get("text_only_acc", 0.0)))
            val_match_cosines.append(float(val_stats.get("match_cosine", 0.0)))
            val_random_cosines.append(float(val_stats.get("random_cosine", 0.0)))
            print(
                f"[{backbone}] epoch {epoch}/{args.epochs} | train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}"
            )

            if metrics_handle is not None:
                row = {
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
                    "match_cosine": float(val_stats.get("match_cosine", 0.0)),
                    "random_cosine": float(val_stats.get("random_cosine", 0.0)),
                    "confusion_matrix": val_stats.get("confusion_matrix", []),
                }
                metrics_handle.write(json.dumps(row) + "\n")

            # Always track best validation accuracy/epoch for reporting; only save checkpoint when requested.
            if va_acc > best_val:
                best_val = va_acc
                best_epoch = epoch
                if args.save_best_model:
                    # Enforce that parent folder exists just in case it was accidentally deleted during the run!
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"img": img_model.state_dict(), "txt": txt_model.state_dict(), "clf": clf.state_dict()}, checkpoint_path)
    finally:
        if metrics_handle is not None:
            metrics_handle.close()

    summary = {
        "backbone": backbone,
        "best_val_acc": float(best_val),
        "best_epoch": int(best_epoch),
        "final_val_acc": float(val_accs[-1]) if val_accs else 0.0,
        "mean_val_acc": float(sum(val_accs) / len(val_accs)) if val_accs else 0.0,
        "final_macro_f1": float(val_f1s[-1]) if val_f1s else 0.0,
        "mean_macro_f1": float(sum(val_f1s) / len(val_f1s)) if val_f1s else 0.0,
        "final_image_only_acc": float(val_image_only_accs[-1]) if val_image_only_accs else 0.0,
        "final_text_only_acc": float(val_text_only_accs[-1]) if val_text_only_accs else 0.0,
        "final_match_cosine": float(val_match_cosines[-1]) if val_match_cosines else 0.0,
        "final_random_cosine": float(val_random_cosines[-1]) if val_random_cosines else 0.0,
        "config_used": cfg,
        "metrics_file": str(metrics_path) if metrics_path else None,
        "checkpoint": str(checkpoint_path) if (best_val > 0 and args.save_best_model) else None,
    }

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id",
            "backbone",
            "best_val_acc",
            "best_epoch",
            "final_val_acc",
            "mean_val_acc",
            "final_macro_f1",
            "mean_macro_f1",
            "final_image_only_acc",
            "final_text_only_acc",
            "final_match_cosine",
            "final_random_cosine",
            "metrics_file",
            "checkpoint",
        ])
        writer.writerow([
            f"part_a_{backbone}",
            backbone,
            f"{best_val:.6f}",
            best_epoch,
            f"{summary['final_val_acc']:.6f}",
            f"{summary['mean_val_acc']:.6f}",
            f"{summary['final_macro_f1']:.6f}",
            f"{summary['mean_macro_f1']:.6f}",
            f"{summary['final_image_only_acc']:.6f}",
            f"{summary['final_text_only_acc']:.6f}",
            f"{summary['final_match_cosine']:.6f}",
            f"{summary['final_random_cosine']:.6f}",
            str(metrics_path) if metrics_path else None,
            str(checkpoint_path) if (best_val > 0 and args.save_best_model) else None,
        ])

    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    if metrics_path:
        print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved summary JSON: {summary_path}")
    if summary["checkpoint"]:
        print(f"Saved best checkpoint: {summary['checkpoint']}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=resolve_default_data_dir())
    parser.add_argument("--backbone", choices=["baseline", "clip"], default="baseline")
    parser.add_argument(
        "--compare-backbones",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run both baseline and clip backbones and rank them. Use --no-compare-backbones for a single backbone.",
    )
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-metrics", action=argparse.BooleanOptionalAction, default=False, help="Save per-epoch metrics.jsonl when enabled.")
    parser.add_argument("--save-best-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to write metrics and checkpoints")
    parser.add_argument(
        "--use-tuning-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, load best hyperparameters from tuning_runs/<latest>/best_configs.json when available",
    )
    parser.add_argument("--clf-hidden-dim", type=int, default=256)
    parser.add_argument("--clf-dropout", type=float, default=0.2)
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

    # Loads the latest tuned hyperparameters from huperparemeter_tuning_ab.py
    tuning = _load_latest_tuning(Path(__file__).resolve().parent / "tuning_runs") if args.use_tuning_best else None
    backbones = ["baseline", "clip"] if args.compare_backbones else [args.backbone]

    print(f"Compare backbones: {args.compare_backbones}")
    print(f"Backbones to run: {backbones}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = args.output_dir or (Path("part_a_runs") / f"part_a_{ts}")
    run_root.mkdir(parents=True, exist_ok=True)

    results = [_run_part_a_backbone(backbone, args, tuning, run_root, device) for backbone in backbones]
    results = sorted(results, key=lambda row: row["best_val_acc"], reverse=True)

    # Saves the results 
    if len(results) > 1:
        summary_csv = run_root / "summary.csv"
        summary_path = run_root / "summary.json"

        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["run_id", "backbone", "best_val_acc", "best_epoch", "metrics_file", "checkpoint"])
            for row in results:
                writer.writerow(
                    [
                        f"part_a_{row['backbone']}",
                        row["backbone"],
                        f"{row['best_val_acc']:.6f}",
                        row["best_epoch"],
                        row["metrics_file"],
                        row["checkpoint"],
                    ]
                )

        with summary_path.open("w", encoding="utf-8") as sf:
            json.dump(results, sf, indent=2)

        print(f"Saved summary CSV: {summary_csv}")
        print(f"Saved summary JSON: {summary_path}")

    # Saves best results
    best = results[0] if results else None
    if best is not None:
        print("\n=== Best Part A Run ===")
        print(f"best_val_acc={best['best_val_acc']:.4f} at epoch {best['best_epoch']} | backbone={best['backbone']}")


if __name__ == "__main__":
    main()
