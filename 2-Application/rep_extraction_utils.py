"""Minimal representation extraction utilities.
- extract_one(...)
- collect_stats(...)
- extract_layer_values(...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
    from io_utils import read_jsonlines, write_jsonlines
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.append(str(_Path(__file__).resolve().parent))
    from io_utils import read_jsonlines, write_jsonlines



@dataclass
class ExtractConfig:
    dataset_jsonl: str
    out_jsonl: str
    check_timestep: int = 10
    hook_names: Sequence[str] = None
    pooling: str = "spatial_max"  # spatial_max | spatial_mean | channel_mean
    use_prompt: bool = False
    image_size: int = 512


def _load_image_tensor(image_path: str, *, image_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = torch.from_numpy(__import__("numpy").array(img)).to(dtype=torch.float32) / 255.0  # [H,W,3]
    x = arr.permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)  # [1,3,H,W] in [0,1]
    return x


def _encode_prompt(pipe, prompt: str, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    # Use pipeline tokenizer/text_encoder directly (works for SD1.x).
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    enc = pipe.text_encoder(input_ids)[0]
    return enc.to(dtype=dtype)


def _pool_feat(x: torch.Tensor, pooling: str) -> torch.Tensor:
    # x: [B,C,H,W] or [B,N,D] or [B,D]
    if x.ndim == 4:
        if pooling == "spatial_max":
            return x.flatten(2).amax(dim=2)  # [B,C]
        if pooling == "spatial_mean":
            return x.flatten(2).mean(dim=2)  # [B,C]
        # fallback
        return x.flatten(1)
    if x.ndim == 3:
        if pooling == "channel_mean":
            return x.mean(dim=1)  # [B,D]
        return x.flatten(1)
    return x


@torch.no_grad()
def extract_one(
    pipe,
    *,
    image_path: str,
    prompt: str = "",
    check_timestep: int = 10,
    hook_names: Sequence[str] = None,
    pooling: str = "spatial_max",
    use_prompt: bool = False,
    image_size: int = 512,
    return_reps: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    """Extract pooled activations + simple stats for one image.

    Fixes common dtype issues by matching UNet dtype end-to-end.
    """
    device = pipe.device.type if hasattr(pipe, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = getattr(pipe.unet, "dtype", torch.float32)

    # --- hooks ---
    feats: Dict[str, torch.Tensor] = {}

    def _make_hook(name: str):
        def _hook(_m, _inp, out):
            if isinstance(out, (tuple, list)):
                out = out[0]
            feats[name] = out
        return _hook

    handles = []
    name_set = set(hook_names)
    for name, module in pipe.unet.named_modules():
        if name in name_set:
            handles.append(module.register_forward_hook(_make_hook(name)))

    # --- forward ---
    x = _load_image_tensor(image_path, image_size=image_size, device=device, dtype=getattr(pipe.vae, "dtype", model_dtype))
    x = (x * 2.0 - 1.0).to(dtype=getattr(pipe.vae, "dtype", model_dtype))  # [-1,1] for VAE

    timestep = torch.tensor([int(check_timestep)], device=device, dtype=torch.long)

    latents = pipe.vae.encode(x).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    latents = latents.to(dtype=model_dtype)

    noise = torch.randn_like(latents, dtype=model_dtype)
    noisy_latents = pipe.scheduler.add_noise(latents, noise, timestep).to(dtype=model_dtype)

    prompt_used = prompt if use_prompt else ""
    enc = _encode_prompt(pipe, prompt_used, device=device, dtype=model_dtype)

    _ = pipe.unet(noisy_latents, timestep, encoder_hidden_states=enc)[0]

    for h in handles:
        h.remove()

    reps: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Dict[str, float]] = {}

    for name in hook_names:
        if name not in feats:
            continue
        v = _pool_feat(feats[name], pooling=pooling)[0].detach().float().cpu()
        reps[name] = v
        l2 = v.norm(p=2).item()
        l4 = v.norm(p=4).item()
        stats[name] = {
            "std": v.std(unbiased=False).item(),
            "l2": l2,
            "l4_over_l2": (l4 / (l2 + 1e-12)),
        }

    if return_reps:
        return reps, stats
    # keep API: first return value exists but may be unused
    return reps, stats



##### Stats computation (jsonl)
@torch.no_grad()
def stats_to_jsonl(
    cfg: ExtractConfig,
    *,
    pipe,
    rows: Optional[Sequence[Mapping]] = None,
) -> List[Mapping]:
    """Loop over a jsonl dataset ({url,prompt}) and write per-sample activation stats.

    Not "too integrated": you pass `pipe`, and optionally pass `rows` directly.
    """
    if rows is None:
        rows = list(read_jsonlines(cfg.dataset_jsonl))

    out_rows = []
    for r in tqdm(rows, desc=f"Extract reps @t={cfg.check_timestep}"):
        url = r.get("url") or r.get("image") or r.get("path")
        prompt = r.get("prompt") or r.get("caption") or r.get("text") or ""

        _, stats = extract_one(
            pipe,
            image_path=url,
            prompt=prompt,
            check_timestep=cfg.check_timestep,
            hook_names=cfg.hook_names,
            pooling=cfg.pooling,
            use_prompt=cfg.use_prompt,
            image_size=cfg.image_size,
            return_reps=False,
        )

        out_rows.append(
            {
                "url": url,
                "prompt": prompt,
                "check_timestep": int(cfg.check_timestep),
                "pooling": cfg.pooling,
                "features": stats,
            }
        )

    Path(cfg.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    write_jsonlines(out_rows, cfg.out_jsonl)
    return out_rows

def extract_layer_values(
    stats_rows: Union[str, Sequence[Mapping]],
    layer: str,
    *,
    key: str = "std",
) -> np.ndarray:
    """Extract a 1D float32 array from per-sample stats.

    Args:
      stats_rows: either a path to `*_stats.jsonl`, or a list of dicts returned by `stats_to_jsonl`.
      layer: hook name to read from `row["features"][layer]`.
      key: which scalar to read (default: "std").

    Returns:
      np.ndarray of shape [N] (float32).
    """
    if isinstance(stats_rows, str):
        stats_rows = list(read_jsonlines(stats_rows))

    vals = []
    for r in stats_rows:
        feats = r.get("features", {})
        if layer in feats and key in feats[layer]:
            vals.append(float(feats[layer][key]))
    return np.asarray(vals, dtype=np.float32)

