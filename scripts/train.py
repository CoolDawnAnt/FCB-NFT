# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Field-Constrained Bargaining NFT (FCB-NFT) trainer for SD3.5-Medium.

The trainer is DiffusionNFT's train_nft_sd3.py with ONE algorithmic change: how the
per-sample scalar optimality o in [0,1] is built from K rewards.

  DiffusionNFT (baseline):  per-prompt centered, globally normalized advantage of a fixed
                            weighted-sum reward, clipped to +-A, o = 0.5 + adv/(2A).
  FCB-NFT (this method):    rewards are only prompt-group CENTERED. Per prompt group of
                            B rollouts:
        C_hat = R~^T R~ / (B-1)                        reward covariance
        M_hat = (1/BL) sum_{i,l} q_il R~_i R~_i^T      conservative NFT field metric
                (q_il = mean-squared velocity residual of the rollout model at a probe)
        a*    = argmax_{a>=0}  sum_k omega_k log([C a]_k / r0 - d_k)
                s.t.  a^T M a <= Delta,   eps_o <= r0 + R~_i a <= 1 - eps_o
        o_i   = r0 + R~_i^T a*
  The disagreement d starts at 0 and is relaxed (d = -tau_d s_ref, tau_d doubling up to a
  cap) only if the strict program is infeasible; if still infeasible the group gets no
  tilt (a = 0, o = r0). Everything downstream of o (implicit positive/negative branches,
  loss, EMA, rollout-anchor schedule) is unchanged from DiffusionNFT.

Method / baseline selection is by environment variables (see run_ocr.sh / run_cmp.sh):
  BNFT_MODE=bargain                        FCB-NFT
  BNFT_MODE=bargain BNFT_FIELD_MODE=reward_only   reward-only bargaining ablation
  BNFT_MODE=baseline BNFT_BASELINE_NORM=raw|globalnorm|groupz   fixed scalarizations
"""

from collections import defaultdict, deque
import os
import sys
import datetime
from concurrent import futures
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from absl import app, flags
import logging
from diffusers import StableDiffusion3Pipeline
import numpy as np
import cvxpy as cp
import flow_grpo.rewards
from flow_grpo.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompt, max_sequence_length)
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


# =====================================================================
# Field-Constrained Bargaining (doc §5-§8)
# =====================================================================


def _symmetrize_psd(mat, ridge_rel=0.0):
    """Symmetrize, clip negative eigenvalues, add a relative trace ridge (doc §8.3)."""
    mat = 0.5 * (mat + mat.T)
    eigval, eigvec = np.linalg.eigh(mat)
    mat = (eigvec * np.clip(eigval, 0.0, None)) @ eigvec.T
    k = mat.shape[0]
    tr = float(np.trace(mat))
    ridge = ridge_rel * (tr / k) if tr > 1e-12 else 1e-10
    return mat + ridge * np.eye(k)


def solve_bargaining_program(C, M, R_tilde, r0, eps_o, field_budget, d, omega, surplus_floor, solver_name,
                             uns=None, sat=None, sat_floor=None, tiny=1e-8, pinned=None):
    """One convex program of doc eq.(10). Returns (a [K], solved: bool).

    max_a  sum_{k in uns} omega_k log(gain_k - d_k + tiny)
    s.t.   a >= 0,  a^T M a <= Delta,
           gain_k - d_k >= surplus_floor_k          (unsaturated channels)
           gain_k >= sat_floor_k                    (saturated channels: PROTECTED,
                                                     excluded from the objective and
                                                     never tau-relaxed)
           eps_o <= r0 + R~ a <= 1 - eps_o          (per rollout sample; box, doc §4.2)

    surplus_floor is a per-channel strictly-positive floor in REWARD UNITS
    (a fraction of the slow global scale). A dimensionless 1e-8 floor lets solver
    tolerance accept zero-gain "solutions" for conflicting rewards instead of
    triggering the explicit infeasibility -> disagreement-relaxation path.
    """
    K = C.shape[0]
    if uns is None:
        uns = list(range(K))
    a = cp.Variable(K, nonneg=True)
    gain = (C @ a) / r0
    surplus = gain - d
    optimality = r0 + R_tilde @ a
    objective = cp.Maximize(cp.sum(cp.multiply(omega[uns], cp.log(surplus[uns] + tiny))))
    constraints = [
        cp.quad_form(a, cp.psd_wrap(M)) <= field_budget,
        surplus[uns] >= surplus_floor[uns],
        optimality >= eps_o,
        optimality <= 1.0 - eps_o,
    ]
    if sat is not None and len(sat) > 0:
        constraints.append(gain[sat] >= sat_floor[sat])
    # Degenerate channels (zero within-group variance: R~_k == 0, so M and C have an
    # exactly-zero row/col) leave a_k unidentifiable -- it enters only through the
    # ridge. That flat direction is what made CLARABEL hard-fail on round-off-level
    # perturbations of otherwise identical data (reward-unit invariance test), after
    # which SCS returned 'optimal_inaccurate' points violating the field budget by
    # up to 40x. Pinning a_k = 0 is semantically exact (no gain, no field cost).
    if pinned is not None and len(pinned) > 0:
        constraints.append(a[pinned] == 0)
    problem = cp.Problem(objective, constraints)
    # CLARABEL can HARD-FAIL (raise SolverError) on ill-conditioned corner cases
    # instead of reporting an infeasible status. A numerical hiccup in one tiny
    # K-variable subproblem must not kill a multi-hour run (observed at epoch 467),
    # so this catch is deliberately NARROW (SolverError only) and loudly logged;
    # the group then falls through the documented relaxation / no-tilt path.
    # SCS (robust first-order) is tried once as a backup before giving up.
    for attempt_solver in (solver_name, "SCS"):
        solver_crashed = False
        try:
            problem.solve(solver=attempt_solver)
        except cp.error.SolverError:
            logger.warning("[bargain] solver %s hard-failed on a group subproblem", attempt_solver)
            solver_crashed = True
        if solver_crashed:
            continue
        if problem.status in ("optimal", "optimal_inaccurate") and a.value is not None:
            a_val = np.maximum(np.asarray(a.value, dtype=np.float64), 0.0)
            # An 'inaccurate' certificate is only accepted if the hard constraints actually
            # hold (relative slack 1e-3); otherwise fall through to the next solver / the
            # explicit relaxation path instead of emitting an over-budget update.
            if problem.status == "optimal_inaccurate":
                cost = float(a_val @ M @ a_val)
                opt = r0 + R_tilde @ a_val
                if cost > field_budget * (1.0 + 1e-3) or opt.min() < eps_o - 1e-3 or opt.max() > 1.0 - eps_o + 1e-3:
                    logger.warning("[bargain] %s returned optimal_inaccurate violating constraints (cost/budget=%.2f); rejected",
                                   attempt_solver, cost / max(field_budget, 1e-300))
                    continue
            return a_val, True
        if problem.status in ("infeasible", "infeasible_inaccurate"):
            return np.zeros(K), False  # genuine infeasibility -> caller relaxes d
    return np.zeros(K), False


def bargain_group(R_raw, q, s_global, bcfg):
    """Full per-prompt-group pipeline (doc §8): stats -> budget -> feasibility
    schedule -> optimality. R_raw [B,K] raw rewards, q [B] mean-squared old-model
    velocity residuals (already averaged over the L probes). Returns (o [B], info).
    """
    B, K = R_raw.shape
    if B < 2:
        return np.full(B, bcfg["r0"]), dict(_EMPTY_INFO, a=np.zeros(K), gain=np.zeros(K), sat=np.zeros(K))
    R_tilde = R_raw - R_raw.mean(axis=0, keepdims=True)  # ONLY centering, no z-score (doc §4.1)
    return _bargain_core(R_tilde, q, s_global, bcfg, c_denom=B - 1)


_EMPTY_INFO = {
    "mode": "no_tilt", "tau_d": 0.0, "field_cost": 0.0, "field_budget": 0.0, "box_sat": 0.0,
    "field_cost_raw": 0.0,
}


def _bargain_core(R_tilde, q, s_global, bcfg, c_denom):
    """Solve the bargaining program on pre-centered rewards R_tilde [N,K] with
    per-sample field residuals q [N]. c_denom = covariance dof (B-1 for a single
    group; N-G for the pooled per-group-centered estimate, doc §8.4 aggregation).
    """
    N, K = R_tilde.shape
    r0 = bcfg["r0"]
    info = dict(_EMPTY_INFO, a=np.zeros(K), gain=np.zeros(K), sat=np.zeros(K))

    C_raw = _symmetrize_psd(R_tilde.T @ R_tilde / max(c_denom, 1))
    # Conservative field metric \bar M (doc eq.15): mean_i q_i R~_i R~_i^T.
    M_raw = _symmetrize_psd((R_tilde.T * q) @ R_tilde / N)

    c_diag = np.diag(C_raw)
    valid = c_diag > 1e-12
    if not valid.any():
        return np.full(N, r0), info

    # Per-channel UNIT normalization for the SOLVER only (the solution is invariant
    # to positive-affine reward rescaling, doc §7.2, so this changes nothing
    # semantically). Late in training group stds can collapse; in raw units the
    # program then becomes ill-conditioned enough to hard-crash CLARABEL (observed
    # at epoch ~467). In z-units all quantities stay O(1). a is mapped back below.
    inv_unit = 1.0 / np.sqrt(np.clip(c_diag, 1e-12, None))
    # Zero-variance channels carry no information (R~_k == 0); they are removed from the
    # z-unit problem (a_k pinned to 0 below) so the solver never sees 1/sqrt(1e-12) units.
    inv_unit = np.where(valid, inv_unit, 0.0)
    C = C_raw * np.outer(inv_unit, inv_unit)
    M = M_raw * np.outer(inv_unit, inv_unit)
    m_trace = float(np.trace(M))
    M = M + max(bcfg["lambda_m"] * m_trace / K, 1e-10) * np.eye(K)  # ridge in normalized units
    R_n = R_tilde * inv_unit
    s_n = s_global * inv_unit

    # SATURATED-CHANNEL PROTECTION: a channel whose within-group spread has
    # collapsed relative to the global scale has nothing left to bargain in this
    # group. Keeping it in the Nash objective demands impossible gains at its
    # ceiling and drives groups into the disagreement relaxation en masse
    # (observed sustained frac_relaxed ~0.25 -> compounding OCR slides, deepest
    # 0.44 at step 636). Saturated channels are excluded from the objective and
    # get a FIXED no-regression constraint (gain >= -sat_reg_tol * s_global)
    # that never participates in the tau_d relaxation. Fully-degenerate channels
    # (zero variance) are treated the same way.
    group_std = np.sqrt(np.clip(c_diag, 0.0, None))
    sat_mask = (~valid) | (group_std < bcfg["sat_std_rel"] * np.clip(s_global, 1e-12, None))
    uns = np.where(~sat_mask)[0].tolist()
    sat = np.where(sat_mask)[0].tolist()
    info["sat"] = sat_mask.astype(np.float64)
    if len(uns) == 0:
        return np.full(N, r0), info  # every channel saturated: nothing to bargain
    sat_floor_n = -bcfg["sat_reg_tol"] * s_n  # protection floor, solver z-units

    # Field budget Delta: match the field cost of familiar SINGLE-reward NFT updates,
    # a_k^single = 1/(2 Z_k) (doc §9.1), median over channels. CRITICAL: Z_k uses the
    # GLOBAL scale (Z_k = adv_clip_max * s_k^global), matching the validated
    # baseline's global_std=True advantage normalization -- NOT the per-group std.
    # The conservative \bar M ≈ mean(q)*C is proportional to the reward covariance,
    # so a per-group-std calibration cancels the group's absolute spread and pushes
    # EVERY group with the same z-unit tilt; saturated groups (spread << global
    # scale) then get reward-model noise amplified into full-strength updates ->
    # observed oversaturation / color-block collapse after ~100 epochs. With global
    # units the group tilt scales ∝ group_std/global_std (baseline behavior);
    # healthy groups (std ≈ global) are unchanged.
    if bcfg["field_mode"] == "reward_only":
        # Ablation: drop the NFT field geometry entirely; unit ball in z-units.
        M = np.eye(K)
        field_budget = bcfg["delta_scale"] / (4.0 * bcfg["adv_clip_max"] ** 2)
    else:
        s_g = np.clip(s_global, 1e-12, None)
        # median over the channels actually being bargained (saturated ones would
        # drag the budget down with their collapsed M_kk)
        singles = np.diag(M_raw)[uns] / (4.0 * bcfg["adv_clip_max"] ** 2 * s_g[uns] ** 2)
        field_budget = float(np.median(singles)) * bcfg["delta_scale"]
    if field_budget <= 0.0:
        return np.full(N, r0), info
    info["field_budget"] = field_budget

    # Feasibility schedule (doc §8.2): strict d=0 first, then explicit relaxation.
    # The strict-surplus floor is expressed via the slow global scale (in z-units).
    surplus_floor = np.maximum(bcfg["surplus_floor_rel"] * s_n, 1e-10)
    d_strict = np.zeros(K)
    pinned = np.where(~valid)[0].tolist()  # zero-variance channels: a_k = 0
    a_n, solved = solve_bargaining_program(
        C, M, R_n, r0, bcfg["eps_o"], field_budget, d_strict, bcfg["omega"], surplus_floor, bcfg["solver"],
        uns=uns, sat=sat, sat_floor=sat_floor_n, pinned=pinned,
    )
    mode, tau_used = "strict", 0.0
    tau_d = bcfg["tau_d_init"]
    while not solved and tau_d <= bcfg["tau_d_max"]:
        d_relaxed = -tau_d * s_n
        a_n, solved = solve_bargaining_program(
            C, M, R_n, r0, bcfg["eps_o"], field_budget, d_relaxed, bcfg["omega"], surplus_floor,
            bcfg["solver"], uns=uns, sat=sat, sat_floor=sat_floor_n, pinned=pinned,
        )
        mode, tau_used = "relaxed", tau_d
        tau_d *= 2.0
    if not solved:
        return np.full(N, r0), info  # explicit no-update certificate (doc §8.2 step 4)

    a = a_n * inv_unit  # back to raw reward units
    o_raw = r0 + R_tilde @ a
    o = np.clip(o_raw, bcfg["eps_o"], 1.0 - bcfg["eps_o"])
    info.update(
        mode=mode, tau_d=tau_used, a=a, gain=(C_raw @ a) / r0,
        field_cost=float(a_n @ M @ a_n), box_sat=float(np.mean(np.abs(o_raw - o) > 1e-9)),
        field_cost_raw=float(a @ M_raw @ a),  # raw units: comparable ACROSS groups
    )
    return o, info


def compute_baseline_optimalities(prompts_all, avg_rewards, adv_clip_max):
    """Standard DiffusionNFT practical mapping (baseline #1 in doc §12): fixed
    weighted-sum reward, per-prompt mean centering, GLOBAL-std normalization
    (= PerPromptStatTracker(global_std=True)), clip to +-A, o = 0.5 + adv/(2A).
    Bit-matches the original trainer's r under adv_mode="all"."""
    prompts = np.array(prompts_all)
    adv = np.empty_like(avg_rewards)
    std = avg_rewards.std() + 1e-4
    for p in np.unique(prompts):
        members = prompts == p
        adv[members] = (avg_rewards[members] - avg_rewards[members].mean()) / std
    adv = np.clip(adv, -adv_clip_max, adv_clip_max)
    return np.clip(adv / adv_clip_max / 2.0 + 0.5, 0.0, 1.0)


def compute_group_optimalities(prompts_all, rewards_all, q_all, s_global, bcfg):
    """Group gathered samples by prompt and run the bargaining pipeline per group.

    prompts_all: list[str] length N;  rewards_all [N,K];  q_all [N].
    Returns (optimality [N], updated s_global [K], diag dict for logging).
    """
    N, K = rewards_all.shape
    optimality = np.full(N, bcfg["r0"], dtype=np.float64)
    _, inverse = np.unique(np.array(prompts_all), return_inverse=True)
    n_groups = int(inverse.max()) + 1

    # Slow global reward scales: units for the disagreement tolerance AND the field
    # budget's global-unit calibration (never a per-sample divisor; doc §7.2).
    # Per-channel std over ALL gathered samples (incl. between-prompt variance),
    # matching the baseline's global_std=True semantics; robust even when every
    # group's WITHIN-group spread saturates to zero.
    group_stds = np.stack(
        [rewards_all[inverse == g].std(axis=0) if (inverse == g).sum() > 1 else np.zeros(K) for g in range(n_groups)]
    )
    epoch_scale = rewards_all.std(axis=0)
    if s_global is None:
        s_global = epoch_scale
    else:
        s_global = bcfg["scale_ema"] * s_global + (1.0 - bcfg["scale_ema"]) * epoch_scale

    infos = []
    for g in range(n_groups):
        members = np.nonzero(inverse == g)[0]
        o_g, info = bargain_group(rewards_all[members], q_all[members], s_global, bcfg)
        optimality[members] = o_g
        infos.append(info)

    modes = np.array([i["mode"] for i in infos])
    diag = {
        "bargain/frac_strict": float(np.mean(modes == "strict")),
        "bargain/frac_relaxed": float(np.mean(modes == "relaxed")),
        "bargain/frac_no_tilt": float(np.mean(modes == "no_tilt")),
        "bargain/tau_d_mean": float(np.mean([i["tau_d"] for i in infos])),
        "bargain/field_budget_mean": float(np.mean([i["field_budget"] for i in infos])),
        "bargain/field_cost_mean": float(np.mean([i["field_cost"] for i in infos])),
        "bargain/box_sat_frac": float(np.mean([i["box_sat"] for i in infos])),
        "bargain/optimality_mean": float(optimality.mean()),
        "bargain/optimality_std": float(optimality.std()),
        "bargain/q_mean": float(q_all.mean()),
        # Saturation monitor: mean within-group spread relative to the global scale.
        # Healthy ≈ 1; -> 0 means groups saturate and tilts shrink accordingly.
        "bargain/group_std_rel": float(
            np.mean(group_stds.mean(axis=0) / np.clip(s_global, 1e-12, None))
        ),
    }
    a_mat = np.stack([i["a"] for i in infos])       # [G,K]
    gain_mat = np.stack([i["gain"] for i in infos])  # [G,K]
    sat_mat = np.stack([i["sat"] for i in infos])    # [G,K]
    diag["bargain/sat_frac"] = float(sat_mat.mean())
    for k, name in enumerate(bcfg["reward_names"]):
        diag[f"bargain/a_{name}"] = float(a_mat[:, k].mean())
        diag[f"bargain/gain_pred_{name}"] = float(gain_mat[:, k].mean())
        diag[f"bargain/s_global_{name}"] = float(s_global[k])
        diag[f"bargain/sat_{name}"] = float(sat_mat[:, k].mean())
    return optimality, s_global, diag


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()

    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        if current_batch_size < len(sample_neg_prompt_embeds):  # Handle last batch
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        else:
            current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
            current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=current_sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    model_type="sd3",
                )

        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir, transformer_ddp, global_step, epoch, rank, world_size, ema,
    transformer_trainable_parameters, config, optimizer, scaler, extra_state,
):
    """Near-perfect-resume checkpoint. COLLECTIVE: every rank must call (per-rank
    RNG states are gathered to rank 0). Layout of checkpoint-{global_step}/:
      lora/           EMA-swapped PEFT save (back-compat deliverable for eval tools)
      adapters.pt     LIVE lora tensors of ALL adapters (default + old) for resume
      ema.pt          EMA shadow state
      optimizer.pt / scaler.pt
      train_state.pt  epoch (data cursor), global_step, per-rank RNG states, and the
                      rank-0 bargaining state (s_global, adaptive-omega buffers,
                      frozen baseline norm scales). WRITTEN LAST -> its presence
                      marks the checkpoint complete for uploaders/resume pickers.
    """
    rng_state = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if world_size > 1:
        gathered_rng = [None] * world_size if is_main_process(rank) else None
        dist.gather_object(rng_state, gathered_rng, dst=0)
    else:
        gathered_rng = [rng_state]
    if not is_main_process(rank):
        return

    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    model_to_save = transformer_ddp.module

    # LIVE adapter tensors (before any EMA swap), covering default AND old.
    adapters = {n: p.detach().cpu() for n, p in model_to_save.named_parameters() if "lora_" in n}
    torch.save(adapters, os.path.join(save_root, "adapters.pt"))

    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    model_to_save.save_pretrained(save_root_lora)
    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)
        torch.save(ema.state_dict(), os.path.join(save_root, "ema.pt"))

    torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

    train_state = {
        "epoch": epoch,
        "global_step": global_step,
        "world_size": world_size,
        "rng": gathered_rng,
        **(extra_state or {}),
    }
    torch.save(train_state, os.path.join(save_root, "train_state.pt"))
    logger.info(f"Saved checkpoint to {save_root} (epoch {epoch})")


def main(_):
    config = FLAGS.config

    # BNFT_MODE: "bargain" (FCB-NFT optimality) | "baseline" (standard DiffusionNFT
    # advantage on a fixed scalarization). Everything else -- rollout, loss, EMA, anchor
    # schedule, logging keys, eval -- is identical, so curves overlay directly.
    bnft_mode = os.environ.get("BNFT_MODE", "bargain").strip().lower()
    assert bnft_mode in ("bargain", "baseline"), f"unknown BNFT_MODE: {bnft_mode}"
    # Baseline scalarizations:
    #   raw        - equal-weight raw reward summation
    #   globalnorm - per-channel scales FROZEN at the first epoch's all-sample std, then sum
    #   groupz     - per-prompt-group z-score per channel, then sum
    baseline_norm = os.environ.get("BNFT_BASELINE_NORM", "raw").strip().lower()
    assert baseline_norm in ("raw", "globalnorm", "groupz"), f"unknown BNFT_BASELINE_NORM: {baseline_norm}"
    # BNFT_BASELINE_WEIGHTS: optional fixed per-channel weights for the scalarizations
    # (coefficient sweeps), name-based, e.g. "ocr=0.25" (unlisted channels = 1).
    _bw_env = os.environ.get("BNFT_BASELINE_WEIGHTS", "").strip()
    _bw_map = dict(kv.split("=") for kv in _bw_env.split(",")) if _bw_env else {}
    _suffix = "_bargain" if bnft_mode == "bargain" else (
        "_baseline" if baseline_norm == "raw" else f"_baseline_{baseline_norm}")
    config.run_name = os.environ.get(
        "WANDB_RUN_NAME", (config.run_name + _suffix) if config.run_name else _suffix.lstrip("_"))
    _default_save = config.save_dir if bnft_mode == "bargain" else (config.save_dir + _suffix.replace("_bargain", ""))
    config.save_dir = os.environ.get("BNFT_SAVE_DIR", _default_save)

    # ===== FCB-NFT knobs (paper defaults; every value can be overridden by env) =====
    # NOTE: ml_collections ConfigDict is locked; keep knobs as plain locals.
    reward_names = list(config.reward_fn.keys())  # alphabetical (ConfigDict order)
    K_rewards = len(reward_names)
    assert set(_bw_map) <= set(reward_names), f"BNFT_BASELINE_WEIGHTS names {list(_bw_map)} not in {reward_names}"
    baseline_weights = np.array([float(_bw_map.get(n, 1.0)) for n in reward_names], dtype=np.float64)
    if _bw_map:
        logger.info("[baseline] fixed weights: %s" % dict(zip(reward_names, baseline_weights)))
    bcfg = {
        "reward_names": reward_names,
        "r0": float(os.environ.get("BNFT_R0", 0.5)),                 # baseline optimality
        "eps_o": float(os.environ.get("BNFT_EPS_O", 0.03)),          # optimality box margin
        "probes": int(os.environ.get("BNFT_PROBES", 1)),             # L field probes per rollout
        "delta_scale": float(os.environ.get("BNFT_DELTA_SCALE", 1.0)),  # x single-reward calibrated budget
        "tau_d_init": float(os.environ.get("BNFT_TAU_D_INIT", 0.01)),   # disagreement relaxation start (x s_ref)
        "tau_d_max": float(os.environ.get("BNFT_TAU_D_MAX", 0.08)),     # relaxation cap
        "surplus_floor_rel": float(os.environ.get("BNFT_SURPLUS_FLOOR_REL", 1e-3)),  # strict-gain floor (x s_ref)
        "sat_std_rel": float(os.environ.get("BNFT_SAT_STD_REL", 0.15)),  # group std below this x s_ref => saturated
        "sat_reg_tol": float(os.environ.get("BNFT_SAT_REG_TOL", 0.02)),  # saturated channels: gain >= -tol x s_ref
        "lambda_m": float(os.environ.get("BNFT_LAMBDA_M", 1e-3)),    # M ridge (x trace/K)
        "scale_ema": float(os.environ.get("BNFT_SCALE_EMA", 0.9)),   # slow reference-scale EMA
        "solver": os.environ.get("BNFT_SOLVER", "CLARABEL"),
        # "reward_only": keep the Nash objective and C but replace the field constraint by
        # the z-unit ball ||a||^2 <= Delta (ablation).
        "field_mode": os.environ.get("BNFT_FIELD_MODE", "conservative"),
        "omega": np.ones(K_rewards, dtype=np.float64),
        "adv_clip_max": float(config.train.adv_clip_max),            # A of the single-reward calibration
    }
    assert bcfg["field_mode"] in ("conservative", "reward_only"), f"unknown BNFT_FIELD_MODE: {bcfg['field_mode']}"

    # ---- adaptive Nash weights (paper Sec. "Adaptive weights") ----
    # eta_k = realized / predicted gain over a window; omega_k ∝ 1/eta_k, clipped. A ceiling
    # gate caps omega_k at 1 while channel k sits within ceil_margin*s_ref of its running
    # maximum (low eta at the ceiling means "no headroom", not "inefficient"); the running
    # maximum decays by ceil_ref_decay per iteration so a single spike does not disable it.
    omega_adapt = bool(int(os.environ.get("BNFT_OMEGA_ADAPT", 1)))
    omega_window = int(os.environ.get("BNFT_OMEGA_WINDOW", 25))
    omega_clip = float(os.environ.get("BNFT_OMEGA_CLIP", 2.0))
    eta_ema_decay = float(os.environ.get("BNFT_ETA_EMA", 0.95))
    ceil_margin = float(os.environ.get("BNFT_CEIL_MARGIN", 0.1))
    ceil_ref_decay = float(os.environ.get("BNFT_CEIL_REF_DECAY", 0.997))
    omega_base = bcfg["omega"].copy()
    reward_means_hist = deque(maxlen=omega_window)  # per-epoch mean raw rewards [K]
    gain_pred_hist = deque(maxlen=omega_window)     # per-epoch mean predicted gains [K]
    eta_ema = None
    reward_hist_max = None                          # per-channel running max of epoch means

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(project=os.environ.get("WANDB_PROJECT", "fcb-nft"), name=config.run_name,
                   config=config.to_dict(), dir=log_dir)
    logger.info(f"\n{config}")
    logger.info("[%s] rewards=%s knobs=%s" % (bnft_mode, reward_names, {k: v for k, v in bcfg.items() if k != "omega"}))

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_3.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device)

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    train_dataset = TextPromptDataset(config.dataset, "train")
    test_dataset = TextPromptDataset(config.dataset, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=train_dataset.collate_fn, pin_memory=True
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    # Bargaining needs per-prompt groups; single-image groups have no within-group
    # statistics (C=M=0) and would silently produce o=r0 everywhere.
    assert config.sample.num_image_per_prompt > 1, "BargainingNFT requires rollout groups (num_image_per_prompt > 1)"

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(device, config.reward_fn)  # Pass device

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    _resume_state = None
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        state_path = os.path.join(config.resume_from, "train_state.pt")
        adapters_path = os.path.join(config.resume_from, "adapters.pt")
        if os.path.exists(state_path) and os.path.exists(adapters_path):
            # Near-perfect resume: LIVE adapters (default + old), optimizer, scaler
            # here; EMA / RNG / rank-0 bargaining state restored after their init
            # points below. first_epoch restores the data cursor exactly (the
            # KRepeat sampler is a deterministic function of seed + epoch).
            _ad = torch.load(adapters_path, map_location="cpu")
            _missing, _unexpected = transformer_ddp.module.load_state_dict(
                {k: v.to(device) for k, v in _ad.items()}, strict=False)
            assert not _unexpected, f"unexpected adapter keys on resume: {_unexpected[:5]}"
            _resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
            global_step = int(_resume_state["global_step"])
            first_epoch = int(_resume_state["epoch"])
            logger.info(f"Perfect resume: epoch={first_epoch} global_step={global_step}")
        else:
            # legacy checkpoints (pre-train_state.pt): EMA-swapped lora only
            lora_path = os.path.join(config.resume_from, "lora")
            if os.path.exists(lora_path):
                transformer_ddp.module.load_adapter(lora_path, adapter_name="default", is_trainable=True)
                transformer_ddp.module.load_adapter(lora_path, adapter_name="old", is_trainable=False)
            _step_str = os.path.basename(config.resume_from).split("-")[-1]
            global_step = int(_step_str) if _step_str.isdigit() else 0
            first_epoch = global_step  # 1 optimizer step per epoch
            logger.info(f"Legacy resume: global_step={global_step}")

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
        if _resume_state is not None:
            _ema_path = os.path.join(config.resume_from, "ema.pt")
            if os.path.exists(_ema_path):
                ema.load_state_dict(torch.load(_ema_path, map_location="cpu"))
                ema.to(device)

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    if _resume_state is None:  # perfect resume already restored the true old adapter
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param

    s_global = None  # slow per-channel reward scale EMA (rank 0 only; solver runs there)
    frozen_norm_scales = None  # baseline globalnorm: per-channel scales frozen at epoch 0

    if _resume_state is not None:
        # rank-0 bargaining/baseline state (harmless no-ops on other ranks)
        s_global = _resume_state.get("s_global")
        frozen_norm_scales = _resume_state.get("frozen_norm_scales")
        eta_ema = _resume_state.get("eta_ema")
        reward_hist_max = _resume_state.get("reward_hist_max")
        reward_means_hist.extend(_resume_state.get("reward_means_hist", []))
        gain_pred_hist.extend(_resume_state.get("gain_pred_hist", []))
        # per-rank RNG (noise / probe timesteps / shuffles continue their streams)
        _rng_all = _resume_state.get("rng")
        if _rng_all is not None and len(_rng_all) == world_size:
            _r = _rng_all[rank]
            torch.set_rng_state(torch.as_tensor(_r["torch"], dtype=torch.uint8))
            torch.cuda.set_rng_state(torch.as_tensor(_r["cuda"], dtype=torch.uint8))
            np.random.set_state(_r["numpy"])
            random.setstate(_r["python"])
        else:
            logger.warning("[resume] RNG states missing or world-size mismatch; streams restart")

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
            )
            prompt_ids = tokenizers[0](
                prompts, padding="max_length", max_length=256, truncation=True, return_tensors="pt"
            ).input_ids.to(device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if i == 0 and epoch % config.save_freq == 0 and not config.debug:
                # COLLECTIVE (RNG gather); rank 0 supplies the bargaining state.
                _extra = {
                    "s_global": s_global,
                    "frozen_norm_scales": frozen_norm_scales,
                    "reward_means_hist": list(reward_means_hist),
                    "gain_pred_hist": list(gain_pred_hist),
                    "eta_ema": eta_ema,
                    "reward_hist_max": reward_hist_max,
                    "bnft_mode": bnft_mode,
                    "baseline_norm": baseline_norm,
                } if is_main_process(rank) else None
                save_ckpt(
                    config.save_dir, transformer_ddp, global_step, epoch, rank, world_size,
                    ema, transformer_trainable_parameters, config, optimizer, scaler, _extra,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[: len(prompts)],
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds[: len(prompts)],
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=config.sample.deterministic,
                        solver=config.sample.solver,
                        model_type="sd3",
                    )

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            # --- Conservative field probe --------------------------------------------
            # q_i = (1/L) sum_l MeanOverLatentDims || v_target - v_old(x_t) ||^2 at
            # renoised probe points on the SAME timestep grid the NFT loss trains on.
            # Uses the "old" (rollout) adapter, i.e. the population baseline v^old of the
            # theory. Rewards are attached later (per group, M_hat = (1/B) sum_i q_i R~_i R~_i^T).
            x0_probe = latents[:, -1].float()
            field_q = torch.zeros(x0_probe.shape[0], device=device, dtype=torch.float32)
            _need_probe = bnft_mode == "bargain" and bcfg["field_mode"] == "conservative"
            with torch.no_grad():
                for _probe in range(bcfg["probes"] if _need_probe else 0):
                    probe_idx = torch.randint(0, timesteps.shape[1], (x0_probe.shape[0],), device=device)
                    t_probe_ms = torch.gather(timesteps, 1, probe_idx.unsqueeze(1)).squeeze(1).float()
                    t_probe = (t_probe_ms / 1000.0).view(-1, *([1] * (x0_probe.ndim - 1)))
                    probe_noise = torch.randn_like(x0_probe)
                    xt_probe = (1 - t_probe) * x0_probe + t_probe * probe_noise
                    v_target_probe = probe_noise - x0_probe  # flow velocity for alpha=1-t, sigma=t
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        v_old_probe = transformer_ddp(
                            hidden_states=xt_probe,
                            timestep=t_probe_ms,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_prompt_embeds,
                            return_dict=False,
                        )[0]
                    field_q += ((v_target_probe - v_old_probe.float()) ** 2).mean(
                        dim=tuple(range(1, x0_probe.ndim))
                    ) / bcfg["probes"]
            transformer_ddp.module.set_adapter("default")

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "field_q": field_q,
                    "rewards_future": rewards_future,  # Store future
                }
            )

        for sample_item in tqdm(
            samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0) for sk in samples_data_list[0][k]}
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][-len(images_to_log) :].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )

        # Gather rewards / field probes / prompt ids across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()
        gathered_field_q = gather_tensor_to_all(collated_samples["field_q"], world_size).numpy().astype(np.float64)
        prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
        prompts_all_decoded = pipeline.tokenizer.batch_decode(prompt_ids_all.cpu().numpy(), skip_special_tokens=True)

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )
            # local per-epoch rollout means (staged sweeps select on this without wandb)
            if not config.debug:
                os.makedirs(config.save_dir, exist_ok=True)
                _rl = os.path.join(config.save_dir, "reward_log.csv")
                _new = not os.path.exists(_rl)
                with open(_rl, "a") as _f:
                    if _new:
                        _f.write("epoch,step," + ",".join(reward_names) + "\n")
                    _f.write(f"{epoch},{global_step}," + ",".join(
                        f"{gathered_rewards_dict[n].mean():.6f}" for n in reward_names) + "\n")

        # =====================================================================
        # Scalar optimality construction, on rank 0 (deterministic single source),
        # broadcast to all ranks.
        #   bargain : field-constrained Nash bargaining over the K reward channels
        #   baseline: standard NFT z-score advantage on the weighted-avg reward
        # Shared diagnostics keep the SAME wandb keys so runs overlay directly.
        # =====================================================================
        samples_per_gpu = collated_samples["timesteps"].shape[0]
        n_total = world_size * samples_per_gpu
        optimality_all = torch.full((n_total,), bcfg["r0"], device=device, dtype=torch.float32)
        if is_main_process(rank):
            rewards_all = np.stack(
                [gathered_rewards_dict[name].astype(np.float64) for name in reward_names], axis=1
            )  # [N, K]
            if bnft_mode == "baseline":
                _, _inv = np.unique(np.array(prompts_all_decoded), return_inverse=True)
                _bw = baseline_weights
                if baseline_norm == "raw":
                    # == gathered "avg" (config.reward_fn weights are all 1.0) when _bw is all-ones
                    scalar_reward = (rewards_all * _bw).sum(axis=1)
                elif baseline_norm == "globalnorm":
                    # fixed global normalization (paper baseline #3): scales frozen
                    # at the FIRST epoch's all-sample std, persisted in checkpoints
                    if frozen_norm_scales is None:
                        frozen_norm_scales = np.clip(rewards_all.std(axis=0), 1e-6, None)
                        logger.info("[baseline] frozen global norm scales: %s" % np.round(frozen_norm_scales, 5))
                    scalar_reward = (rewards_all / frozen_norm_scales * _bw).sum(axis=1)
                else:  # groupz (paper baseline #4): per-group per-channel z-score, then sum
                    z_all = np.empty_like(rewards_all)
                    for _g in range(int(_inv.max()) + 1):
                        _m = _inv == _g
                        _grp = rewards_all[_m]
                        z_all[_m] = (_grp - _grp.mean(axis=0)) / (_grp.std(axis=0) + 1e-4)
                    scalar_reward = (z_all * _bw).sum(axis=1)
                optimality_np = compute_baseline_optimalities(
                    prompts_all_decoded, scalar_reward, bcfg["adv_clip_max"]
                )
                _gstds = np.stack(
                    [rewards_all[_inv == g].std(axis=0) for g in range(int(_inv.max()) + 1)]
                )
                _escale = rewards_all.std(axis=0)
                s_global = _escale if s_global is None else (
                    bcfg["scale_ema"] * s_global + (1.0 - bcfg["scale_ema"]) * _escale)
                bargain_diag = {
                    "bargain/optimality_mean": float(optimality_np.mean()),
                    "bargain/optimality_std": float(optimality_np.std()),
                    "bargain/group_std_rel": float(
                        np.mean(_gstds.mean(axis=0) / np.clip(s_global, 1e-12, None))
                    ),
                }
                logger.info(
                    "[baseline] o_mean=%.3f o_std=%.3f std_rel=%.3f"
                    % (
                        bargain_diag["bargain/optimality_mean"],
                        bargain_diag["bargain/optimality_std"],
                        bargain_diag["bargain/group_std_rel"],
                    )
                )
            else:
                # eta_k = realized reward movement over the last W epochs divided by
                # the cumulative predicted gain issued in those epochs (both raw
                # units; cross-channel leakage via reward correlations is absorbed
                # into eta). omega is updated BEFORE this epoch's solve.
                reward_means_now = rewards_all.mean(axis=0)
                reward_hist_max = reward_means_now if reward_hist_max is None else np.maximum(
                    reward_hist_max * ceil_ref_decay, reward_means_now)
                if (
                    omega_adapt
                    and len(reward_means_hist) == omega_window
                    and len(gain_pred_hist) == omega_window
                ):
                    realized = reward_means_now - reward_means_hist[0]
                    predicted = np.sum(np.stack(gain_pred_hist), axis=0)
                    eta_raw = np.clip(realized / np.clip(predicted, 1e-8, None), 0.02, 5.0)
                    eta_ema = eta_raw if eta_ema is None else (
                        eta_ema_decay * eta_ema + (1.0 - eta_ema_decay) * eta_raw)
                    inv_eta = 1.0 / eta_ema
                    omega_ad = np.clip(
                        K_rewards * inv_eta / inv_eta.sum(), 1.0 / omega_clip, omega_clip)
                    if s_global is not None:  # ceiling gate (see knob comment above)
                        at_ceiling = (reward_hist_max - reward_means_now) < ceil_margin * np.clip(
                            s_global, 1e-12, None)
                        omega_ad = np.where(at_ceiling, np.minimum(omega_ad, 1.0), omega_ad)
                    bcfg["omega"] = omega_base * omega_ad
                optimality_np, s_global, bargain_diag = compute_group_optimalities(
                    prompts_all_decoded, rewards_all, gathered_field_q, s_global, bcfg
                )
                reward_means_hist.append(reward_means_now)
                gain_pred_hist.append(
                    np.array([bargain_diag[f"bargain/gain_pred_{n}"] for n in reward_names])
                )
                if omega_adapt:
                    for _k, _name in enumerate(reward_names):
                        bargain_diag[f"bargain/omega_{_name}"] = float(bcfg["omega"][_k])
                        if eta_ema is not None:
                            bargain_diag[f"bargain/eta_{_name}"] = float(eta_ema[_k])
                        if s_global is not None:
                            bargain_diag[f"bargain/at_ceiling_{_name}"] = float(
                                (reward_hist_max[_k] - reward_means_now[_k])
                                < ceil_margin * max(float(s_global[_k]), 1e-12)
                            )
                logger.info(
                    "[bargain] strict=%.2f relaxed=%.2f no_tilt=%.2f o_mean=%.3f o_std=%.3f std_rel=%.3f"
                    % (
                        bargain_diag["bargain/frac_strict"],
                        bargain_diag["bargain/frac_relaxed"],
                        bargain_diag["bargain/frac_no_tilt"],
                        bargain_diag["bargain/optimality_mean"],
                        bargain_diag["bargain/optimality_std"],
                        bargain_diag["bargain/group_std_rel"],
                    )
                )
            optimality_all = torch.from_numpy(optimality_np).float().to(device)
            wandb.log(bargain_diag, step=global_step)
        if world_size > 1:
            dist.broadcast(optimality_all, src=0)

        collated_samples["optimality"] = optimality_all.reshape(world_size, samples_per_gpu)[rank]

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        del collated_samples["field_q"]

        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        filtered_samples = collated_samples

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples["timesteps"].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {k: v[perm] for k, v in filtered_samples.items()}

            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])

                if config.sample.guidance_scale > 1.0:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:current_micro_batch_size], train_sample_batch["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [
                            train_neg_pooled_prompt_embeds[:current_micro_batch_size],
                            train_sample_batch["pooled_prompt_embeds"],
                        ]
                    )
                else:
                    embeds = train_sample_batch["prompt_embeds"]
                    pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt,
                                timestep=train_sample_batch["timesteps"][:, j_idx],
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=train_sample_batch["timesteps"][:, j_idx],
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt,
                                        timestep=train_sample_batch["timesteps"][:, j_idx],
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        return_dict=False,
                                    )[0]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}
                    # BargainingNFT: r IS the bargained optimality (already in
                    # [eps_o, 1-eps_o]); no advantage clipping / z-score mapping.
                    # Cached per rollout group: every gradient step of this epoch
                    # reuses the same solved coefficients (doc §8.4).
                    r = torch.clamp(train_sample_batch["optimality"], 0.0, 1.0)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    # Keep the base trainer's constant loss scale (adv_clip_max) so the
                    # validated learning rate transfers unchanged.
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()
                    loss_terms["optimality_mean"] = r.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

    # final checkpoint at the target step (periodic saves stop at the last multiple of save_freq)
    if not config.debug:
        _extra = {
            "s_global": s_global,
            "frozen_norm_scales": frozen_norm_scales,
            "reward_means_hist": list(reward_means_hist),
            "gain_pred_hist": list(gain_pred_hist),
            "eta_ema": eta_ema,
            "reward_hist_max": reward_hist_max,
            "bnft_mode": bnft_mode,
            "baseline_norm": baseline_norm,
        } if is_main_process(rank) else None
        save_ckpt(
            config.save_dir, transformer_ddp, global_step, config.num_epochs, rank, world_size,
            ema, transformer_trainable_parameters, config, optimizer, scaler, _extra,
        )

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
