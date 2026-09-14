#!/usr/bin/env bash
# Launch one training run.
#
#   bash run.sh <suite> <method> [extra --config.* overrides]
#
#   suite   ocr            OCR + PickScore + HPSv2 + CLIPScore            (OCR suite)
#           cmp            OCR + JPEG-compressibility + PickScore + HPSv2  (compressibility suite)
#   method  fcb            FCB-NFT (ours)
#           rewardonly     reward-only bargaining ablation (field constraint -> ||a||^2 <= Delta)
#           rawsum         DiffusionNFT on the equal-weight raw reward sum
#           globalnorm     DiffusionNFT on the sum of globally normalised rewards
#           groupz         DiffusionNFT on the sum of per-group z-scored rewards
#           single_<name>  single-reward DiffusionNFT specialist,
#                          <name> in ocr | pickscore | hpsv2 | clipscore | jpeg_compressibility
#
# Environment knobs (all optional):
#   NPROC=8            number of GPUs on this node (4 / 8 / 16 are pre-configured, see below)
#   NUM_EPOCHS=1000    total outer iterations (= optimizer steps)
#   RESUME_FROM=<dir>  logs/<run>/checkpoints/checkpoint-N to resume from (exact data cursor / RNG / EMA)
#   WANDB_MODE=offline to skip wandb login
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SUITE="${1:?usage: run.sh <ocr|cmp> <fcb|rewardonly|rawsum|globalnorm|groupz|single_<reward>> [overrides]}"
METHOD="${2:?usage: run.sh <ocr|cmp> <fcb|rewardonly|rawsum|globalnorm|groupz|single_<reward>> [overrides]}"
shift 2

case "$SUITE" in
    ocr) CONFIG_NAME=ocr_multi ;;
    cmp) CONFIG_NAME=cmp_multi ;;
    *) echo "unknown suite: $SUITE"; exit 1 ;;
esac
case "$METHOD" in
    fcb)        export BNFT_MODE=bargain ;;
    rewardonly) export BNFT_MODE=bargain BNFT_FIELD_MODE=reward_only ;;
    rawsum)     export BNFT_MODE=baseline BNFT_BASELINE_NORM=raw ;;
    globalnorm) export BNFT_MODE=baseline BNFT_BASELINE_NORM=globalnorm ;;
    groupz)     export BNFT_MODE=baseline BNFT_BASELINE_NORM=groupz ;;
    single_*)   export BNFT_MODE=baseline BNFT_BASELINE_NORM=raw; CONFIG_NAME="$METHOD" ;;
    *) echo "unknown method: $METHOD"; exit 1 ;;
esac

# One outer iteration = 48 prompt groups x 12 images = 576 rollouts, one optimizer step,
# for every GPU count. Constraint: NPROC*TBS % 12 == 0 and NPROC*TBS*NBATCH == 576.
NPROC="${NPROC:-8}"
case "$NPROC" in
    16) TBS=6;  NBATCH=6 ;;
    8)  TBS=9;  NBATCH=8 ;;
    4)  TBS=9;  NBATCH=16 ;;
    *)  TBS="${TBS:?set TBS and NBATCH for NPROC=$NPROC (NPROC*TBS % 12 == 0, NPROC*TBS*NBATCH == 576)}"
        NBATCH="${NBATCH:?}" ;;
esac

RESUME_FLAG=""
[ -n "${RESUME_FROM:-}" ] && RESUME_FLAG="--config.resume_from=$RESUME_FROM"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node="$NPROC" --master_port="${MASTER_PORT:-29563}" scripts/train.py \
    --config "config/nft.py:$CONFIG_NAME" \
    $RESUME_FLAG \
    --config.sample.train_batch_size="$TBS" \
    --config.train.batch_size="$TBS" \
    --config.sample.num_batches_per_epoch="$NBATCH" \
    --config.train.gradient_accumulation_steps="$NBATCH" \
    --config.sample.num_image_per_prompt=12 \
    --config.train.timestep_fraction=0.28 \
    --config.train.learning_rate=4e-4 \
    --config.num_epochs="${NUM_EPOCHS:-1000}" \
    --config.eval_freq=25 \
    --config.save_freq=50 \
    "$@"
