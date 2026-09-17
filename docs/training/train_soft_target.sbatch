#!/bin/bash
#SBATCH --job-name=palm_soft_target
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Set these paths before submission or pass them through sbatch --export.
TRAIN_DIR="${TRAIN_DIR:?Set TRAIN_DIR to the training FASTA-shard directory}"
VAL_DIR="${VAL_DIR:?Set VAL_DIR to the validation FASTA-shard directory}"
IPR_TSV="${IPR_TSV:?Set IPR_TSV to the InterPro label table}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/palm_soft_target_epoch8_${SLURM_JOB_ID}}"
GPUS="${GPUS:-2}"

mkdir -p logs "${OUTPUT_DIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

torchrun --standalone --nproc-per-node="${GPUS}" training/train_soft_target.py \
  --train-dir "${TRAIN_DIR}" \
  --val-dir "${VAL_DIR}" \
  --ipr-tsv "${IPR_TSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 8 \
  --batch-size 24 \
  --max-len 4096 \
  --workers 10 \
  --learning-rate 1e-4 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.05 \
  --lora-last-n-layers 4 \
  --seed 1234

