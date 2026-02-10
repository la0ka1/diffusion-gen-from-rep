"""Minimal image generation + storage utilities.
- `load_sd_pipe(...)` loads a Stable Diffusion pipeline (DDIM scheduler, safety checker off).
- `generate_and_store(...)` generates images and writes PNGs + a jsonl manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch

try:
    from diffusers import DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
except Exception as e:  # pragma: no cover
    DDIMScheduler = UNet2DConditionModel = StableDiffusionPipeline = None
    _DIFFUSERS_IMPORT_ERROR = e

try:
    from io_utils import write_jsonlines
except ModuleNotFoundError:  # pragma: no cover
    # allow running as a standalone file
    import sys as _sys

    _sys.path.append(str(Path(__file__).resolve().parent))
    from io_utils import write_jsonlines


__all__ = ["load_sd_pipe", "generate_and_store"]


def load_sd_pipe(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    *,
    unet_id: Optional[str] = None,
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load a Stable Diffusion pipeline with DDIM scheduler.

    Notes:
      - CPU forces float32 for correctness.
      - If `torch_dtype` is None on CUDA, we default to float16.
      - Safety checker is disabled.
    """
    if StableDiffusionPipeline is None:
        raise ImportError(
            "diffusers is required. Install with: pip install diffusers transformers accelerate"
        ) from _DIFFUSERS_IMPORT_ERROR

    if device.startswith("cpu"):
        torch_dtype = torch.float32
    elif torch_dtype is None:
        torch_dtype = torch.float16

    if unet_id is not None:
        unet = UNet2DConditionModel.from_pretrained(unet_id, torch_dtype=torch_dtype)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            unet=unet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


@torch.no_grad()
def generate_and_store(
    pipe,
    prompts: Sequence[str],
    *,
    out_root: str = "samples",
    run_name: str = "run",
    seed: int = 0,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    batch_size: int = 1,
) -> str:
    """Generate images and store as PNG + jsonl manifest.

    Writes:
      - {out_root}/{run_name}/{idx:06d}.png
      - {out_root}/{run_name}_all.jsonl   (each line: {url, prompt})

    Returns:
      Path to the jsonl file (as a string).
    """
    batch_size = max(1, int(batch_size))

    out_root = Path(out_root)
    img_dir = out_root / run_name
    img_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator(device=pipe.device).manual_seed(int(seed))

    rows = []
    idx = 0
    for s in range(0, len(prompts), batch_size):
        batch = list(prompts[s : s + batch_size])

        images = pipe(
            batch,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            height=int(height),
            width=int(width),
            generator=g,
        ).images

        for p, im in zip(batch, images):
            fp = img_dir / f"{idx:06d}.png"
            im.save(fp)
            rows.append({"url": fp.as_posix(), "prompt": str(p)})
            idx += 1

    jsonl_path = out_root / f"{run_name}_all.jsonl"
    write_jsonlines(rows, str(jsonl_path))
    return str(jsonl_path)
