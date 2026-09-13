#!/usr/bin/env bash
# LoRA fine-tune pi0 on the sim dataset.  Sized for a single RTX 3090 (24 GB).
#
#   bash train_pi0_lora.sh                     # defaults below
#   DATA_ROOT=data/my_run STEPS=20000 bash train_pi0_lora.sh
#
# What keeps it inside 24 GB:
#   * LoRA (PEFT) on the 300M action expert's q/v projections + the small
#     action/state projection layers; the 3B PaliGemma VLM stays frozen.
#     (lerobot's default pi0 PEFT targets -- override with PEFT_TARGETS.)
#   * bf16 weights, gradient checkpointing, batch 8.
# If you still OOM: BATCH_SIZE=4 GRAD_ACCUM=2.
set -euo pipefail
cd "$(dirname "$0")"

DATA_ROOT="${DATA_ROOT:-data/so101_pick_cube}"
REPO_ID="${REPO_ID:-local/so101_sim_pick_cube}"
BASE="${BASE:-lerobot/pi0_base}"            # pi0 base weights on the HF Hub
OUT="${OUT:-outputs/pi0_so101_lora}"
STEPS="${STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-1e-4}"                            # LoRA likes ~4x the full-FT lr (2.5e-5)
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
WANDB="${WANDB:-false}"

EXTRA=()
if [[ -n "${PEFT_TARGETS:-}" ]]; then EXTRA+=("--peft.target_modules=${PEFT_TARGETS}"); fi

python -m lerobot.scripts.lerobot_train \
  --policy.path="$BASE" \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps=500 \
  --policy.scheduler_decay_steps="$STEPS" \
  --peft.method_type=LORA \
  --peft.r="$LORA_R" \
  --peft.lora_alpha="$LORA_ALPHA" \
  "${EXTRA[@]}" \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATA_ROOT" \
  --dataset.image_transforms.enable=true \
  --batch_size="$BATCH_SIZE" \
  --accelerator.gradient_accumulation.steps="$GRAD_ACCUM" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --log_freq=50 \
  --num_workers=4 \
  --output_dir="$OUT" \
  --job_name=pi0_so101_lora \
  --wandb.enable="$WANDB" \
  "$@"
