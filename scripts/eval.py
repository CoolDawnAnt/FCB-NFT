"""Held-out evaluation on the OCR test prompts (paired across models).

Protocol:
  * full ocr test split (1017 prompts), one image per prompt
  * per-prompt FIXED init latents (seed = SEED + prompt_idx, CPU generator) and
    deterministic flow sampling -> identical noise for every model, so per-prompt
    reward deltas are paired across checkpoints
  * 40 inference steps, CFG-free (guidance 1.0), 512px -- identical to the trainer's
    in-training eval call
  * all five rewards (ocr / pickscore / hpsv2 / clipscore / jpeg_compressibility) are
    scored for every model, so specialists fill the off-diagonal entries of the tables

Run (one model per invocation; any world size, sharding is strided):
  torchrun --nproc_per_node=8 scripts/eval.py --tag fcb --lora logs/ocr_multi/checkpoints/checkpoint-950/lora
  torchrun --nproc_per_node=8 scripts/eval.py --tag base --lora none        # SD3.5-M init row
Rank outputs land in --out_dir as {tag}.rank{r}.json; rank 0 merges them into {tag}.json.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
from peft import PeftModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

import flow_grpo.rewards
from diffusers import StableDiffusion3Pipeline
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARDS = {"ocr": 1.0, "pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0, "jpeg_compressibility": 1.0}
LATENT_SHAPE = (16, 64, 64)  # SD3 @ 512px: 16 ch, 512/8 spatial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lora", required=True, help="lora dir (checkpoint-N/lora, EMA weights) or 'none'")
    ap.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "eval_results"))
    ap.add_argument("--dataset_dir", default=os.path.join(REPO_ROOT, "dataset", "ocr"))
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_steps", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N prompts")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.dataset_dir, "test.txt")) as f:
        prompts_all = [line.strip() for line in f.readlines()]
    if args.limit:
        prompts_all = prompts_all[: args.limit]

    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium")
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=torch.float16)
    pipeline.text_encoder_2.to(device, dtype=torch.float16)
    pipeline.text_encoder_3.to(device, dtype=torch.float16)
    transformer = pipeline.transformer
    if args.lora != "none":
        transformer = PeftModel.from_pretrained(transformer, args.lora)
        transformer = transformer.merge_and_unload()
    pipeline.transformer = transformer.to(device)
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(disable=True)
    for m in (pipeline.vae, pipeline.text_encoder, pipeline.text_encoder_2,
              pipeline.text_encoder_3, pipeline.transformer):
        m.requires_grad_(False)
        m.eval()

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    reward_fn = flow_grpo.rewards.multi_score(device, REWARDS)

    with torch.no_grad():
        neg_embed, neg_pooled = encode_prompt(text_encoders, tokenizers, [""], 128)
    neg_embed, neg_pooled = neg_embed.to(device), neg_pooled.to(device)

    my_idxs = list(range(rank, len(prompts_all), world))
    results = {"idx": [], **{k: [] for k in REWARDS}, "avg": []}
    for lo in range(0, len(my_idxs), args.batch_size):
        idxs = my_idxs[lo: lo + args.batch_size]
        prompts = [prompts_all[i] for i in idxs]
        with torch.no_grad():
            embeds, pooled = encode_prompt(text_encoders, tokenizers, prompts, 128)
        embeds, pooled = embeds.to(device), pooled.to(device)
        latents = torch.stack([
            torch.randn(LATENT_SHAPE, generator=torch.Generator("cpu").manual_seed(args.seed + i))
            for i in idxs
        ]).to(device=device, dtype=embeds.dtype)

        with torch.autocast("cuda", dtype=torch.float16):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=embeds,
                    pooled_prompt_embeds=pooled,
                    negative_prompt_embeds=neg_embed[: len(idxs)].expand(len(idxs), -1, -1),
                    negative_pooled_prompt_embeds=neg_pooled[: len(idxs)].expand(len(idxs), -1),
                    latents=latents,
                    num_inference_steps=args.num_steps,
                    guidance_scale=1.0,
                    output_type="pt",
                    height=512,
                    width=512,
                    noise_level=0.7,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )
        rewards, _ = reward_fn(images, prompts, [{} for _ in prompts], only_strict=False)
        results["idx"].extend(idxs)
        for k in list(REWARDS) + ["avg"]:
            vals = torch.as_tensor(rewards[k]).detach().float().cpu().numpy()
            results[k].extend(vals.tolist())
        if rank == 0:
            print(f"[eval:{args.tag}] {lo + len(idxs)}/{len(my_idxs)} per-rank prompts done", flush=True)

    rank_path = os.path.join(args.out_dir, f"{args.tag}.rank{rank}.json")
    with open(rank_path, "w") as f:
        json.dump(results, f)
    dist.barrier()

    if rank == 0:
        merged = {"idx": [], **{k: [] for k in REWARDS}, "avg": []}
        for r in range(world):
            with open(os.path.join(args.out_dir, f"{args.tag}.rank{r}.json")) as f:
                part = json.load(f)
            for k in merged:
                merged[k].extend(part[k])
        order = np.argsort(merged["idx"])
        final = {
            "tag": args.tag, "lora": args.lora, "seed": args.seed,
            "num_steps": args.num_steps, "n": len(order),
            "per_prompt": {k: np.asarray(merged[k])[order].tolist() for k in list(REWARDS) + ["avg"]},
        }
        final["mean"] = {k: float(np.mean(final["per_prompt"][k])) for k in list(REWARDS) + ["avg"]}
        with open(os.path.join(args.out_dir, f"{args.tag}.json"), "w") as f:
            json.dump(final, f)
        print(f"[eval:{args.tag}] n={final['n']} means=" +
              " ".join(f"{k}={final['mean'][k]:.4f}" for k in REWARDS), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
