from __future__ import annotations

import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def autocast_ctx(use_amp: bool):
    if not use_amp:
        return nullcontext()
    try:
        return torch.amp.autocast("cuda", enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(use_amp: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def forward_embeddings(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    output = model(imgs)
    if isinstance(output, tuple):
        return output[0]
    return output


def extract_embeddings(loader, model: nn.Module, device: torch.device, use_amp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    embeds_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast_ctx(use_amp and device.type == "cuda"):
                embeds = forward_embeddings(model, imgs)
            embeds_list.append(embeds.float().cpu().numpy())
            labels_list.append(labels.numpy())
    return np.concatenate(embeds_list), np.concatenate(labels_list)


def model_summary_text(
    model: nn.Module,
    arcface: nn.Module | None = None,
    config: dict[str, Any] | None = None,
) -> str:
    base_model = unwrap_model(model)
    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    lines = [
        f"model_class: {base_model.__class__.__name__}",
        f"parameters_total: {total}",
        f"parameters_trainable: {trainable}",
    ]
    if arcface is not None:
        lines.append(f"arcface_parameters: {sum(p.numel() for p in arcface.parameters())}")
    if config:
        for key in ["run_id", "model_name", "use_msff", "use_angular_mask", "mask_type", "polar_height", "polar_width"]:
            lines.append(f"{key}: {config.get(key)}")
    return "\n".join(lines) + "\n"
