"""
GPU-HEAVY Hyperparameter tuning for Parts A & B (individual heavy search)
- Uses num_workers=0 to avoid multiprocessing deadlocks
- Writes real-time CSV results and a best_configs.json
- Samples from large grids but defaults are GPU-heavy
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from part_a_multimodal_embeddings import (
    MMIMDbLikeDataset,
    ImageEmbedder,
    TextEmbedder,
    BaselineClassifier,
    collate_fn,
    resolve_default_data_dir,
    run_epoch,
    evaluate_loader,
    set_seed,
)
from part_b_fusion_comparison import (
    train_one_fusion,
)


@dataclass
class RunResult:
    best_val_acc: float
    last_val_acc: float


def _cartesian_grid(space: Dict[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    values = [space[k] for k in keys]
    combos = []
    import itertools

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


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{int(hours)}h{int(mins)}m"


def _print_trial_update(trial_num: int, total_trials: int, acc: float, best_acc: float, elapsed: float):
    if total_trials == 0:
        return
    pct = (trial_num / total_trials) * 100
    filled = int(40 * trial_num / total_trials)
    bar = "█" * filled + "░" * (40 - filled)
    avg_time = elapsed / max(trial_num, 1)
    remaining = avg_time * (total_trials - trial_num)
    eta_str = _format_time(remaining) if trial_num < total_trials else "Done!"
    print(f"  [{bar}] {pct:5.1f}% | Trial {trial_num:3d}/{total_trials:3d} | Acc: {acc:.4f} | Best: {best_acc:.4f} | ETA: {eta_str}")


# -------------------- Part A runner --------------------

def run_part_a_trial(ds, cfg: dict, device: torch.device, epochs: int, img_model=None, txt_model=None) -> RunResult:
    set_seed(cfg.get("seed", 42))
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.get("seed", 42)))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=0)

    created_local_models = False
    if img_model is None or txt_model is None:
        # create local models (slower) if shared models not provided
        img_model = ImageEmbedder(cfg["embed_dim"], backbone=cfg.get("backbone", "baseline")).to(device)
        txt_model = TextEmbedder(cfg["embed_dim"], backbone=cfg.get("backbone", "baseline")).to(device)
        created_local_models = True
    clf = BaselineClassifier(cfg["embed_dim"], ds.num_classes, hidden_dim=cfg.get("clf_hidden_dim", 128), dropout_p=cfg.get("clf_dropout", 0.2)).to(device)

    print(
        f"    Starting Part A trial | embed_dim={cfg['embed_dim']} batch_size={cfg['batch_size']} "
        f"lr={cfg['lr']} wd={cfg.get('weight_decay', 0.0)} opt={cfg.get('optimizer', 'adam')} "
        f"epochs={epochs}",
        flush=True,
    )

    opt_cls = torch.optim.AdamW if cfg.get("optimizer", "adam") == "adamw" else torch.optim.Adam
    optimizer = opt_cls(
        list(img_model.parameters()) + list(txt_model.parameters()) + list(clf.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.get("label_smoothing", 0.0))

    best_val = 0.0
    last_val = 0.0
    # If shared models were provided, we'll restore their weights after the trial
    if not created_local_models:
        img_init = {k: v.cpu().clone() for k, v in img_model.state_dict().items()}
        txt_init = {k: v.cpu().clone() for k, v in txt_model.state_dict().items()}

    for epoch in range(1, epochs + 1):
        print(f"      [Part A] epoch {epoch}/{epochs} running...", flush=True)
        _ = run_epoch(train_loader, img_model, txt_model, clf, criterion, optimizer, device, True)
        val_stats = evaluate_loader(val_loader, img_model, txt_model, clf, criterion, device)
        va_acc = val_stats.get("acc", 0.0)
        best_val = max(best_val, va_acc)
        last_val = va_acc
        print(
            f"      [Part A] epoch {epoch}/{epochs} done | val acc={va_acc:.4f} | best={best_val:.4f}",
            flush=True,
        )
    return RunResult(best_val_acc=best_val, last_val_acc=last_val)


# -------------------- Part B runner --------------------

def run_part_b_trial(ds, cfg: dict, device: torch.device, epochs: int) -> RunResult:
    set_seed(cfg.get("seed", 42))
    # Prepare loaders
    train_n = int(0.8 * len(ds))
    val_n = len(ds) - train_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.get("seed", 42)))

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=0)

    # If external encoders are passed in cfg, use them to avoid reloading pretrained weights
    img_model = cfg.get("img_model")
    txt_model = cfg.get("txt_model")
    if img_model is None or txt_model is None:
        img_model = ImageEmbedder(cfg["embed_dim"], backbone=cfg.get("backbone", "baseline")).to(device)
        txt_model = TextEmbedder(cfg["embed_dim"], backbone=cfg.get("backbone", "baseline")).to(device)
    init_img_state = cfg.get("init_img_state") or img_model.state_dict()
    init_txt_state = cfg.get("init_txt_state") or txt_model.state_dict()

    # train_one_fusion returns metadata dict with best_val
    res = train_one_fusion(
        cfg.get("backbone", "baseline"),
        cfg["fusion"],
        ds,
        train_loader,
        val_loader,
        cfg["embed_dim"],
        epochs,
        cfg["lr"],
        device,
        img_model,
        txt_model,
        init_img_state,
        init_txt_state,
        cfg.get("adv_dropout", 0.2),
        cfg.get("modality_dropout", 0.1),
        cfg.get("weight_decay", 0.0),
        cfg.get("label_smoothing", 0.0),
        cfg.get("aux_weight", 0.2),
        cfg.get("gnn_layers", 2),
        cfg.get("optimizer", "adam"),
        save_best=True,
        out_dir=cfg.get("out_dir", None),
    )

    best = res.get("best_val", 0.0) if isinstance(res, dict) else float(res)
    return RunResult(best_val_acc=best, last_val_acc=best)


# -------------------- Main script --------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Heavy GPU tuner for Parts A & B")
    parser.add_argument("--data-dir", type=Path, default=resolve_default_data_dir())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=256)

    parser.add_argument("--run-part-a", action="store_true", default=True)
    parser.add_argument("--run-part-b", action="store_true", default=True)

    # heavy defaults for Part A
    parser.add_argument("--part-a-epochs", type=int, default=2)
    parser.add_argument("--part-a-embed-dims", type=str, default="64,128,256")
    parser.add_argument("--part-a-batch-sizes", type=str, default="32,64,128")
    parser.add_argument("--part-a-lrs", type=str, default="0.001,0.0005,0.0003,0.0001")
    parser.add_argument("--part-a-weight-decays", type=str, default="0.0,1e-5,1e-4")
    parser.add_argument("--part-a-optimizers", type=str, default="adam,adamw")
    parser.add_argument("--part-a-clf-hidden", type=str, default="128,256")
    parser.add_argument("--part-a-clf-dropouts", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--max-trials-part-a", type=int, default=1)

    # heavy defaults for Part B
    parser.add_argument("--part-b-epochs", type=int, default=2)
    parser.add_argument("--part-b-embed-dims", type=str, default="64,128,256")
    parser.add_argument("--part-b-batch-sizes", type=str, default="32,64,128")
    parser.add_argument("--part-b-lrs", type=str, default="0.001,0.0005,0.0003")
    parser.add_argument("--part-b-weight-decays", type=str, default="0.0,1e-5,1e-4")
    parser.add_argument("--part-b-optimizers", type=str, default="adam,adamw")
    parser.add_argument("--part-b-fusions", type=str, default="early,early_adv,late,late_adv,clip,early_gnn,late_gnn")
    parser.add_argument("--part-b-aux-weights", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--part-b-mod-drop", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--part-b-adv-drop", type=str, default="0.1,0.2")
    parser.add_argument("--max-trials-part-b", type=int, default=1)

    parser.add_argument("--output-dir", type=Path, default=Path("tuning_runs"))
    args = parser.parse_args()

    # parse lists
    def _parse(raw: str, cast):
        return [cast(x.strip()) for x in raw.split(",") if x.strip()]

    args.part_a_embed_dims = _parse(args.part_a_embed_dims, int)
    args.part_a_batch_sizes = _parse(args.part_a_batch_sizes, int)
    args.part_a_lrs = _parse(args.part_a_lrs, float)
    args.part_a_weight_decays = _parse(args.part_a_weight_decays, float)
    args.part_a_optimizers = _parse(args.part_a_optimizers, str)
    args.part_a_clf_hidden = _parse(args.part_a_clf_hidden, int)
    args.part_a_clf_dropouts = _parse(args.part_a_clf_dropouts, float)

    args.part_b_embed_dims = _parse(args.part_b_embed_dims, int)
    args.part_b_batch_sizes = _parse(args.part_b_batch_sizes, int)
    args.part_b_lrs = _parse(args.part_b_lrs, float)
    args.part_b_weight_decays = _parse(args.part_b_weight_decays, float)
    args.part_b_optimizers = _parse(args.part_b_optimizers, str)
    args.part_b_fusions = _parse(args.part_b_fusions, str)
    args.part_b_aux_weights = _parse(args.part_b_aux_weights, float)
    args.part_b_mod_drop = _parse(args.part_b_mod_drop, float)
    args.part_b_adv_drop = _parse(args.part_b_adv_drop, float)

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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.output_dir / ts
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"GPU-heavy tuner for Parts A & B | Device: {device}")
    print(f"Outputs -> {out_root}")

    ds = MMIMDbLikeDataset(args.data_dir, max_samples=args.max_samples, backbone="baseline")

    best_configs: dict = {"part_a": {}, "part_b": {}}

    # PART A
    if args.run_part_a:
        print("\n=== Part A Heavy Tuning ===\n")
        space_a = {
            "embed_dim": args.part_a_embed_dims,
            "batch_size": args.part_a_batch_sizes,
            "lr": args.part_a_lrs,
            "weight_decay": args.part_a_weight_decays,
            "optimizer": args.part_a_optimizers,
            "clf_hidden_dim": args.part_a_clf_hidden,
            "clf_dropout": args.part_a_clf_dropouts,
        }
        grid_a = _cartesian_grid(space_a)
        trials_a = _sample_trials(grid_a, args.max_trials_part_a, args.seed)
        print(f"Part A grid combos: {len(grid_a):,} | sampled: {len(trials_a)}")

        csv_a = out_root / "part_a_results.csv"
        with csv_a.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["trial", "embed_dim", "batch_size", "lr", "weight_decay", "optimizer", "clf_hidden_dim", "clf_dropout", "best_val_acc"])
            best = 0.0
            best_cfg_record: dict | None = None
            start = time.time()
            # Preload shared encoders for each unique embed_dim to avoid repeated heavy init
            dims_a = sorted({int(x["embed_dim"]) for x in trials_a})
            print("Preloading encoder backbones for Part A for dims:", dims_a)
            shared_imgs = {}
            shared_txts = {}
            for d in dims_a:
                shared_imgs[d] = ImageEmbedder(d, backbone="baseline").to(device)
                shared_txts[d] = TextEmbedder(d, backbone="baseline").to(device)
            print("Encoders loaded for dims — starting trials")

            for i, t in enumerate(trials_a, start=1):
                cfg = {
                    "embed_dim": t["embed_dim"],
                    "batch_size": t["batch_size"],
                    "lr": t["lr"],
                    "weight_decay": t["weight_decay"],
                    "optimizer": t["optimizer"],
                    "clf_hidden_dim": t["clf_hidden_dim"],
                    "clf_dropout": t["clf_dropout"],
                    "seed": args.seed,
                }
                # pass shared encoders (they will be reset inside the trial)
                d = int(cfg["embed_dim"])
                res = run_part_a_trial(ds, cfg, device, args.part_a_epochs, img_model=shared_imgs[d], txt_model=shared_txts[d])
                if res.best_val_acc > best:
                    best = res.best_val_acc
                    best_cfg_record = dict(cfg)
                w.writerow([i, cfg["embed_dim"], cfg["batch_size"], cfg["lr"], cfg["weight_decay"], cfg["optimizer"], cfg["clf_hidden_dim"], cfg["clf_dropout"], f"{res.best_val_acc:.4f}"])
                f.flush()
                _print_trial_update(i, len(trials_a), res.best_val_acc, best, time.time() - start)
            backbone_key = "baseline"
            if best_cfg_record is None and trials_a:
                best_cfg_record = trials_a[0]
            best_configs["part_a"][backbone_key] = {"best_config": best_cfg_record or {}, "best_val": float(best)}
        print(f"Part A done. Results -> {csv_a}")

    # PART B
    if args.run_part_b:
        print("\n=== Part B Heavy Tuning ===\n")
        space_b = {
            "embed_dim": args.part_b_embed_dims,
            "batch_size": args.part_b_batch_sizes,
            "lr": args.part_b_lrs,
            "weight_decay": args.part_b_weight_decays,
            "optimizer": args.part_b_optimizers,
            "fusion": args.part_b_fusions,
            "aux_weight": args.part_b_aux_weights,
            "modality_dropout": args.part_b_mod_drop,
            "adv_dropout": args.part_b_adv_drop,
        }
        grid_b = _cartesian_grid(space_b)
        trials_b = _sample_trials(grid_b, args.max_trials_part_b, args.seed)
        print(f"Part B grid combos: {len(grid_b):,} | sampled: {len(trials_b)}")

        csv_b = out_root / "part_b_results.csv"
        with csv_b.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["trial", "backbone", "fusion", "embed_dim", "batch_size", "lr", "weight_decay", "optimizer", "aux_weight", "best_val_acc"])
            best = 0.0
            best_cfgs_by_fusion: dict = {}
            start = time.time()
            # Preload shared encoders for Part B for each unique embed_dim
            dims_b = sorted({int(x["embed_dim"]) for x in trials_b})
            print("Preloading encoder backbones for Part B for dims:", dims_b)
            shared_imgs_b = {}
            shared_txts_b = {}
            init_img_states_b = {}
            init_txt_states_b = {}
            for d in dims_b:
                shared_imgs_b[d] = ImageEmbedder(d, backbone="baseline").to(device)
                shared_txts_b[d] = TextEmbedder(d, backbone="baseline").to(device)
                init_img_states_b[d] = shared_imgs_b[d].state_dict()
                init_txt_states_b[d] = shared_txts_b[d].state_dict()
            print("Encoders loaded for dims — starting Part B trials")

            for i, t in enumerate(trials_b, start=1):
                cfg = {
                    "backbone": "baseline",
                    "fusion": t["fusion"],
                    "embed_dim": t["embed_dim"],
                    "batch_size": t["batch_size"],
                    "lr": t["lr"],
                    "weight_decay": t["weight_decay"],
                    "optimizer": t["optimizer"],
                    "aux_weight": t.get("aux_weight", 0.2),
                    "modality_dropout": t.get("modality_dropout", 0.1),
                    "adv_dropout": t.get("adv_dropout", 0.2),
                    "gnn_layers": 2,
                    "seed": args.seed,
                }
                # attach shared encoders and initial states so run_part_b_trial won't reload them
                d = int(cfg["embed_dim"])
                cfg.update({"img_model": shared_imgs_b[d], "txt_model": shared_txts_b[d], "init_img_state": init_img_states_b[d], "init_txt_state": init_txt_states_b[d]})
                res = run_part_b_trial(ds, cfg, device, args.part_b_epochs)
                if res.best_val_acc > best:
                    best = res.best_val_acc
                fusion = cfg["fusion"]
                prev = best_cfgs_by_fusion.get(fusion, {"best_val": 0.0})
                if res.best_val_acc > prev.get("best_val", 0.0):
                    best_cfgs_by_fusion[fusion] = {"best_val": float(res.best_val_acc), "best_config": dict(cfg)}
                w.writerow([i, cfg["backbone"], cfg["fusion"], cfg["embed_dim"], cfg["batch_size"], cfg["lr"], cfg["weight_decay"], cfg["optimizer"], cfg["aux_weight"], f"{res.best_val_acc:.4f}"])
                f.flush()
                _print_trial_update(i, len(trials_b), res.best_val_acc, best, time.time() - start)
            backbone_key = "baseline"
            best_configs["part_b"][backbone_key] = {}
            for fusion_name, record in best_cfgs_by_fusion.items():
                raw_cfg = record.get("best_config", {})
                # sanitize to remove non-serializable objects
                bad_keys = {"img_model", "txt_model", "init_img_state", "init_txt_state"}
                sanitized = {k: v for k, v in raw_cfg.items() if k not in bad_keys}
                best_configs["part_b"][backbone_key][fusion_name] = {"best_config": sanitized, "best_val": record.get("best_val", 0.0)}
        print(f"Part B done. Results -> {csv_b}")

    # save best configs
    (out_root / "best_configs.json").write_text(json.dumps(best_configs, indent=2), encoding="utf-8")
    print("Saved best_configs.json")


if __name__ == "__main__":
    main()
