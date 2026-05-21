#!/bin/bash

# Full experiment suite: Parts A–D with reasonable runtime (~30–45 min on GPU)

set -e  # Exit on any error
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Running all multimodal experiments (A–D)"
echo "=========================================="
echo ""

# Common settings
MAX_SAMPLES=512
BATCH_SIZE=32
SEED=42

# Part A: Baseline encoders + supervised learning 20
echo "[Part A] Running supervised baseline with both backbones..."
python part_a_multimodal_embeddings.py \
  --max-samples $MAX_SAMPLES \
  --epochs 20 \
  --batch-size $BATCH_SIZE \
  --seed $SEED \
  --compare-backbones \
  --output-dir part_a_runs/full_run \
  --save-metrics \
  --save-best-model
echo "✓ Part A complete"
echo ""

# Plot Part A metrics
python plot_metrics.py A part_a_runs/full_run part_a_runs/full_run/part_a_metrics.png || true
echo "Generated Part A plot: part_a_runs/full_run/part_a_metrics.png"
echo ""

# Part B: Fusion strategy comparison 20
echo "[Part B] Running fusion strategy comparison..."
python part_b_fusion_comparison.py \
  --max-samples $MAX_SAMPLES \
  --epochs 20 \
  --batch-size $BATCH_SIZE \
  --seed $SEED \
  --compare-backbones \
  --output-dir part_b_runs/full_run \
  --save-best-model \
  --save-run-metrics
echo "✓ Part B complete"
echo ""

# Plot Part B metrics
python plot_metrics.py B part_b_runs/full_run part_b_runs/full_run/part_b_metrics.png || true
echo "Generated Part B plot: part_b_runs/full_run/part_b_metrics.png"
echo ""

# Part C: Semi-supervised learning 20
echo "[Part C] Running semi-supervised learning sweep..."
python part_c_semisupervised.py \
  --max-samples $MAX_SAMPLES \
  --epochs 20 \
  --batch-size $BATCH_SIZE \
  --seed $SEED \
  --compare-configs \
  --labeled-ratios "0.05,0.1,0.2,0.5,1.0" \
  --pseudo-thresholds "0.9" \
  --consistency-weights "0.5" \
  --strong-drop-probs "0.15" \
  --output-dir part_c_runs/full_run \
  --save-metrics \
  --save-best-model
echo "✓ Part C complete"
echo ""

# Plot Part C metrics
python plot_metrics.py C part_c_runs/full_run part_c_runs/full_run/part_c_metrics.png || true
echo "Generated Part C plot: part_c_runs/full_run/part_c_metrics.png"
echo ""

# Part D: Self-supervised pretraining + fine-tuning 10 25
echo "[Part D] Running self-supervised pretraining + fine-tuning..."
python part_d_selfsupervised.py \
  --max-samples $MAX_SAMPLES \
  --pretrain-epochs 20 \
  --finetune-epochs 20 \
  --batch-size $BATCH_SIZE \
  --seed $SEED \
  --output-dir part_d_runs/full_run \
  --save-metrics \
  --save-checkpoint
echo "✓ Part D complete"
echo ""

# Plot Part D metrics
python plot_metrics.py D part_d_runs/full_run part_d_runs/full_run/part_d_metrics.png || true
echo "Generated Part D plot: part_d_runs/full_run/part_d_metrics.png"
echo ""

echo "=========================================="
echo "✅ All experiments complete!"
echo "=========================================="
echo ""
echo "Results summary:"
echo "  Part A: part_a_runs/full_run/"
echo "  Part B: part_b_runs/full_run/"
echo "  Part C: part_c_runs/full_run/"
echo "  Part D: part_d_runs/full_run/"
echo ""
