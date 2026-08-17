# SPDX-License-Identifier: Apache-2.0
"""Compatibility shims for upstream qwen-tts."""

from __future__ import annotations

from functools import wraps
import inspect
import threading
from typing import Any, Callable

import torch

_APPLY_LOCK = threading.Lock()
_PATCHED_FLAG = "_sglang_omni_qwen_tts_compat_patched"


def _compute_default_rope_parameters(
    config: Any,
    device: torch.device | None = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple[torch.Tensor, float]:
    del seq_len, layer_type
    base = getattr(config, "rope_theta", getattr(config, "default_theta", 10000.0))
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None)
    if head_dim is None:
        head_dim = config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, 1.0


def _patch_mask_helper(masking_utils: Any, name: str) -> None:
    """Adapt qwen-tts's legacy mask-helper keyword names when needed."""
    original = getattr(masking_utils, name, None)
    if original is None or getattr(original, _PATCHED_FLAG, False):
        return

    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return

    parameters = signature.parameters
    if "input_embeds" in parameters or "inputs_embeds" not in parameters:
        return
    accepts_cache_position = "cache_position" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )

    @wraps(original)
    def mask_helper_compat(*args: Any, **kwargs: Any) -> Any:
        if "input_embeds" in kwargs:
            if "inputs_embeds" in kwargs:
                raise TypeError(
                    f"{name} received both input_embeds and inputs_embeds"
                )
            kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
        if not accepts_cache_position:
            kwargs.pop("cache_position", None)
        return original(*args, **kwargs)

    setattr(mask_helper_compat, _PATCHED_FLAG, True)
    setattr(masking_utils, name, mask_helper_compat)


def apply_qwen_tts_transformers_compatibility_patches() -> None:
    """Patch Transformers APIs expected by qwen-tts."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.utils import generic

    try:
        from transformers import masking_utils
    except ImportError:
        masking_utils = None

    with _APPLY_LOCK:
        ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)

        if masking_utils is not None:
            _patch_mask_helper(masking_utils, "create_causal_mask")
            _patch_mask_helper(masking_utils, "create_sliding_window_causal_mask")

        current = generic.check_model_inputs
        if getattr(current, _PATCHED_FLAG, False):
            return

        try:
            signature = inspect.signature(current)
        except (TypeError, ValueError):
            return

        params = list(signature.parameters.values())
        needs_func_arg = (
            len(params) == 1
            and params[0].default is inspect.Parameter.empty
            and params[0].kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
        if not needs_func_arg:
            return

        original = current

        def check_model_inputs_compat(
            func: Callable[..., Any] | None = None,
        ) -> Callable[..., Any]:
            if func is None:

                def decorator(inner: Callable[..., Any]) -> Callable[..., Any]:
                    return original(inner)

                return decorator
            return original(func)

        check_model_inputs_compat.__name__ = getattr(
            original, "__name__", "check_model_inputs"
        )
        check_model_inputs_compat.__doc__ = getattr(original, "__doc__", None)
        setattr(check_model_inputs_compat, _PATCHED_FLAG, True)
        generic.check_model_inputs = check_model_inputs_compat
