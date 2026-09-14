"""Experiment configs for FCB-NFT on SD3.5-Medium (derived from DiffusionNFT's config/nft.py).

    ocr_multi      OCR suite:             ocr + pickscore + hpsv2 + clipscore
    cmp_multi      compressibility suite: ocr + jpeg_compressibility + pickscore + hpsv2
    single_<name>  one single-reward DiffusionNFT specialist per reward (normalized-gain denominators)

Batch topology, sampler steps, learning rate etc. are set by the launch scripts (run_*.sh) so
that one outer iteration is always 48 prompt groups x 12 images regardless of the GPU count.
"""
import importlib.util
import os

_spec = importlib.util.spec_from_file_location("base", os.path.join(os.path.dirname(__file__), "base.py"))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR_SUITE = {"ocr": 1.0, "pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0}
CMP_SUITE = {"ocr": 1.0, "jpeg_compressibility": 1.0, "pickscore": 1.0, "hpsv2": 1.0}


def get_config(name):
    if name.startswith("single_"):
        return _make({name[len("single_"):]: 1.0}, name)
    return globals()[name]()


def _make(reward_fn, name):
    config = base.get_config()
    config.base_model = "sd3"
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(REPO_ROOT, "dataset", "ocr")
    config.prompt_fn = "general_ocr"
    config.resolution = 512
    config.sample.num_steps = 25
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1.0
    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7
    config.sample.test_batch_size = 16
    config.train.beta = 0.0001
    config.train.adv_mode = "all"
    config.beta = 0.1          # NFT implicit-branch coefficient
    config.decay_type = 2      # conservative rollout-anchor schedule (official OCR choice)
    config.reward_fn = reward_fn
    config.run_name = f"nft_sd3_{name}"
    config.save_dir = os.path.join(REPO_ROOT, "logs", name)
    return config


def ocr_multi():
    return _make(dict(OCR_SUITE), "ocr_multi")


def cmp_multi():
    return _make(dict(CMP_SUITE), "cmp_multi")
