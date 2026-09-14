# FCB-NFT

Minimal reference implementation of **Field-Constrained Bargaining NFT
(FCB-NFT)** for multi-reward fine-tuning of Stable Diffusion 3.5 Medium.
The repository contains the two main experiment suites from the paper
(OCR and JPEG-compressibility) plus the fixed-scalarization baselines.

FCB-NFT keeps the DiffusionNFT trainer unchanged and only replaces how the
scalar optimality `o ∈ [0,1]` of each rollout is built from `K` rewards:
per prompt group it estimates the reward covariance `Ĉ` and the
conservative NFT field matrix `M̂`, solves a small convex Nash-bargaining
program for the coefficient vector `a*` under the field budget
`aᵀ M̂ a ≤ Δ`, and sets `o_i = r₀ + R̃_iᵀ a*`.

The code deliberately uses standard PyTorch, Diffusers, PEFT, and
`torchrun`. Model and reward checkpoints are downloaded from Hugging Face.

## Install

Python 3.10+, CUDA, and a recent PyTorch installation are required.
SD3.5 Medium is gated, so accept its Hugging Face license and authenticate
first.

```bash
git clone <repository-url>
cd FCB-NFT
python -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
huggingface-cli login
```

Everything else is downloaded on first use from the Hub
(`stabilityai/stable-diffusion-3.5-medium`, PickScore, CLIP, HPSv2.1,
PaddleOCR). Logging goes to Weights & Biases (`wandb login`, or
`WANDB_MODE=offline`). The OCR prompt set ships under `dataset/ocr/`.

The released settings are memory intensive. We recommend 8 GPUs with at
least 80 GB each. For a small local smoke run:

```bash
WANDB_MODE=offline NUM_EPOCHS=2 bash run.sh ocr fcb --config.debug=True
```

## Train

Both suites train for 1000 outer iterations. Every iteration samples
48 prompt groups × 12 images (576 rollouts) and takes one optimizer step.

OCR suite (`ocr + pickscore + hpsv2 + clipscore`):

```bash
bash run.sh ocr fcb            # FCB-NFT
bash run.sh ocr rawsum         # equal-weight raw sum
bash run.sh ocr globalnorm     # sum after frozen global normalisation
bash run.sh ocr groupz         # sum of per-group z-scores
bash run.sh ocr rewardonly     # reward-only bargaining ablation
bash run.sh ocr single_ocr     # single-reward specialist
                               # (also: pickscore / hpsv2 / clipscore)
```

Compressibility suite (`ocr + jpeg_compressibility + pickscore + hpsv2`):

```bash
bash run.sh cmp fcb
bash run.sh cmp rawsum
bash run.sh cmp globalnorm
bash run.sh cmp groupz
bash run.sh cmp rewardonly
bash run.sh cmp single_jpeg_compressibility
```

`NPROC` selects the GPU count (4 / 8 / 16 are pre-configured with the same
576-rollout iteration). `NUM_EPOCHS` sets the total iterations.
`RESUME_FROM=logs/<run>/checkpoints/checkpoint-N` resumes exactly.
Checkpoints land in `logs/.../checkpoints/checkpoint-<step>/`; the
`lora/` subdirectory holds the EMA LoRA used for evaluation.

Method hyper-parameters default to the paper values inside
`scripts/train.py` and can be overridden with `BNFT_*` environment
variables listed at the top of that file.

## Evaluate

Held-out evaluation on all 1017 OCR test prompts, with the same
per-prompt initial noise across models (paired comparison), 40
deterministic steps, no CFG, 512 px:

```bash
torchrun --nproc_per_node=8 scripts/eval.py --tag base --lora none
torchrun --nproc_per_node=8 scripts/eval.py --tag fcb \
  --lora logs/ocr_multi/checkpoints/checkpoint-950/lora
```

Results are written to `eval_results/<tag>.json`. The paper's normalised
gain is `(R_k(model) − R_k(base)) / (R_k(specialist_k) − R_k(base))`.

## Layout

```
run.sh                 launcher (suite × method → env + torchrun)
scripts/train.py       DiffusionNFT trainer + FCB-NFT optimality
scripts/eval.py        paired held-out evaluation
config/nft.py          suite definitions
flow_grpo/             reward models and SD3 sampling utilities
dataset/ocr/           OCR prompts (train / test)
```

## Acknowledgements

The trainer, sampler, and reward code build on
[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) and
[Flow-GRPO](https://github.com/yifan123/flow_grpo), both Apache-2.0.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`).
