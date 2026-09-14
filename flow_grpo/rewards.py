# Reward functions used by the two FCB-NFT suites. Trimmed from Flow-GRPO / DiffusionNFT
# (Apache-2.0): OCR, PickScore, HPSv2, CLIPScore and DDPO's JPEG compressibility.
# All model checkpoints are pulled from the HuggingFace Hub on first use.
from PIL import Image
import io
import numpy as np
import torch


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    # DDPO definition kept verbatim: reward = -(JPEG kB) / 500
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from flow_grpo.hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def multi_score(device, score_dict):
    """Evaluate every reward in `score_dict` ({name: weight}) and return per-reward
    scores plus their weighted sum under key "avg"."""
    score_functions = {
        "ocr": ocr_score,
        "pickscore": pickscore_score,
        "jpeg_compressibility": jpeg_compressibility,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
    }
    score_fns = {}
    for score_name in score_dict:
        fn = score_functions[score_name]
        score_fns[score_name] = fn(device) if "device" in fn.__code__.co_varnames else fn()

    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}
        for score_name, weight in score_dict.items():
            scores, _ = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]
        score_details["avg"] = total_scores
        return score_details, {}

    return _fn
