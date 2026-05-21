# Option 7 - Multimodal Architectures 

This is a clean restart based on the structure style used in Assignment 8.

## Files

- `part_a_multimodal_embeddings.py`
  - Build text and image embeddings
  - Choose `baseline` or `clip`
  - Train a simple multimodal classifier
- `part_b_fusion_comparison.py`
  - Compare early fusion vs late fusion vs gnn early vs gnn late vs clip
  - Optionally compare baseline vs CLIP too
- `part_c_semisupervised.py`
  - Add pseudo-labeling and consistency training
- `part_d_selfsupervised.py`
  - Add contrastive pretraining and transfer
- `hyperparameter_tuning_ab.py`
  - Full hyperparameter tuning for Part A and Part B
  - Supports all Part B fusion methods
  - Saves per-trial logs and best configs for final experiments
- `hyperparameter_tuning_cd.py`
  - Full hyperparameter tuning for Part C and Part D

## Suggested run order

1. Part A: verify dataset + embeddings work.
2. Part B: compare early and late fusion.
3. Part C: add semi-supervised loop.
4. Part D: add self-supervised pretraining.

## Quick commands

```powershell
pip install -r requirements.txt
python part_a_multimodal_embeddings.py --backbone baseline --max-samples 256 --epochs 1
python part_a_multimodal_embeddings.py --backbone clip --max-samples 256 --epochs 1
python part_b_fusion_comparison.py --no-compare-backbones --backbone baseline --max-samples 256 --epochs 1
python part_b_fusion_comparison.py --compare-backbones --max-samples 256 --epochs 1
python part_c_semisupervised.py --max-samples 256 --epochs 1
python part_d_selfsupervised.py --max-samples 256 --pretrain-epochs 1 --finetune-epochs 1
```

## Full tuning (Part A + Part B)

Run full grid tuning for both backbones and all fusion methods:

```powershell
python hyperparameter_tuning_ab.py \
  --compare-backbones \
  --run-part-a --run-part-b \
  --fusion-methods early late clip early_gnn late_gnn \
  --max-samples -1 \
  --epochs 6 \
  --seeds 42,1337,2026
```

Outputs are saved under `tuning_runs/<timestamp>/`:
- `all_trials_summary.csv`: all trial metrics and configs
- `best_configs.json`: best config per backbone/method
- `run_args.json`: full CLI config used for reproducibility
- `report.md`: readable ranked summary and best-per-method configs

Useful controls:
- `--max-trials-part-a 0 --max-trials-part-b 0`: run full grids (default)
- Set max trials to cap search if needed, e.g. `--max-trials-part-b 150`
- Customize search spaces, e.g. `--embed-dims 64,128,256 --lrs 0.001,0.0003,0.0001`
- Control printed ranking depth with `--top-k-summary 5`

By default, scripts auto-detect the local dataset by checking candidate directories.
The dataset must contain:
- `IMDB_four_genre_larger_plot_description.csv` (Metadata in root)
- `IMDB four_genre_posters/` (Directory with image posters)

The fallback search looks primarily in the parent directory (`Semester assignment Deep Neutral networks`).

## Extra report questions

- Which fusion strategy is more stable when training data is reduced?
- What pseudo-label confidence threshold gives best tradeoff?
- Does contrastive pretraining improve low-data classification?
- How robust is each model when text or image modality is noisy/missing?

# Full bash run

bash run_all_experiments.sh