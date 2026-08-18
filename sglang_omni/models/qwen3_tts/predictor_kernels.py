# SPDX-License-Identifier: Apache-2.0
"""Optional CUDA kernels for the Qwen3-TTS residual-code predictor."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on runtime image
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _gather_codec_embedding_and_add_kernel(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        token_stride,
        embedding_stride,
        gathered_stride,
        accumulated_stride,
        hidden_size: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < hidden_size
        token_id = tl.load(token_ids + row * token_stride)
        values = tl.load(
            embedding_weight + token_id * embedding_stride + offsets,
            mask=mask,
        )
        accumulated_offsets = accumulated + row * accumulated_stride + offsets
        gathered_offsets = gathered + row * gathered_stride + offsets
        current = tl.load(accumulated_offsets, mask=mask)
        tl.store(gathered_offsets, values, mask=mask)
        tl.store(accumulated_offsets, current + values, mask=mask)

    @triton.jit
    def _store_predictor_kv_cache_kernel(
        key,
        value,
        key_cache,
        value_cache,
        key_stride_batch,
        key_stride_head,
        key_stride_dim,
        value_stride_batch,
        value_stride_head,
        value_stride_dim,
        key_cache_stride_batch,
        key_cache_stride_head,
        key_cache_stride_dim,
        value_cache_stride_batch,
        value_cache_stride_head,
        value_cache_stride_dim,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        batch = row // num_kv_heads
        head = row % num_kv_heads
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < head_dim
        key_offsets = (
            key
            + batch * key_stride_batch
            + head * key_stride_head
            + offsets * key_stride_dim
        )
        value_offsets = (
            value
            + batch * value_stride_batch
            + head * value_stride_head
            + offsets * value_stride_dim
        )
        key_cache_offsets = (
            key_cache
            + batch * key_cache_stride_batch
            + head * key_cache_stride_head
            + offsets * key_cache_stride_dim
        )
        value_cache_offsets = (
            value_cache
            + batch * value_cache_stride_batch
            + head * value_cache_stride_head
            + offsets * value_cache_stride_dim
        )
        tl.store(key_cache_offsets, tl.load(key_offsets, mask=mask), mask=mask)
        tl.store(value_cache_offsets, tl.load(value_offsets, mask=mask), mask=mask)

    @triton.jit
    def _store_predictor_kv_and_expand_gqa_first_token_kernel(
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        query_stride_batch,
        query_stride_head,
        query_stride_dim,
        key_stride_batch,
        key_stride_head,
        key_stride_dim,
        value_stride_batch,
        value_stride_head,
        value_stride_dim,
        key_cache_stride_batch,
        key_cache_stride_head,
        key_cache_stride_dim,
        value_cache_stride_batch,
        value_cache_stride_head,
        value_cache_stride_dim,
        output_stride_batch,
        output_stride_head,
        output_stride_dim,
        num_kv_heads: tl.constexpr,
        num_kv_groups: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Cache K/V and materialize the one-token GQA attention output.

        With exactly one key/value token, scaled dot-product attention has a
        scalar softmax of one. Its output is therefore that value vector for
        every query head in the matching GQA group. The score calculation is
        retained only to preserve the reference's non-finite-value behavior.
        """

        row = tl.program_id(0)
        block = tl.program_id(1)
        batch = row // num_kv_heads
        kv_head = row % num_kv_heads
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < head_dim

        key_offsets = (
            key
            + batch * key_stride_batch
            + kv_head * key_stride_head
            + offsets * key_stride_dim
        )
        value_offsets = (
            value
            + batch * value_stride_batch
            + kv_head * value_stride_head
            + offsets * value_stride_dim
        )
        key_values = tl.load(key_offsets, mask=mask)
        value_values = tl.load(value_offsets, mask=mask)

        key_cache_offsets = (
            key_cache
            + batch * key_cache_stride_batch
            + kv_head * key_cache_stride_head
            + offsets * key_cache_stride_dim
        )
        value_cache_offsets = (
            value_cache
            + batch * value_cache_stride_batch
            + kv_head * value_cache_stride_head
            + offsets * value_cache_stride_dim
        )
        tl.store(key_cache_offsets, key_values, mask=mask)
        tl.store(value_cache_offsets, value_values, mask=mask)

        for group_idx in tl.static_range(0, num_kv_groups):
            query_head = kv_head * num_kv_groups + group_idx
            query_offsets = (
                query
                + batch * query_stride_batch
                + query_head * query_stride_head
                + offsets * query_stride_dim
            )
            query_values = tl.load(query_offsets, mask=mask)
            score = tl.sum(query_values * key_values, axis=0)
            score = score * 0.08838834764831843
            is_finite = (
                (score == score) & (score > -float("inf")) & (score < float("inf"))
            )
            # CUDA's one-element softmax returns zero for a -inf score and
            # NaN for +inf or NaN. Keep that edge behavior rather than silently
            # turning it into a valid value-vector copy.
            invalid_probability = tl.where(score == -float("inf"), 0.0, float("nan"))
            output_values = tl.where(
                is_finite, value_values, value_values * invalid_probability
            )
            output_offsets = (
                output
                + batch * output_stride_batch
                + query_head * output_stride_head
                + offsets * output_stride_dim
            )
            tl.store(output_offsets, output_values, mask=mask)

    @triton.jit
    def _store_predictor_kv_and_expand_gqa_cache_kernel(
        key,
        value,
        key_cache,
        value_cache,
        key_stride_batch,
        key_stride_head,
        key_stride_dim,
        value_stride_batch,
        value_stride_head,
        value_stride_dim,
        key_cache_stride_batch,
        key_cache_stride_head,
        key_cache_stride_dim,
        value_cache_stride_batch,
        value_cache_stride_head,
        value_cache_stride_dim,
        num_kv_heads: tl.constexpr,
        num_kv_groups: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Store one K/V token directly in query-head GQA cache layout."""

        row = tl.program_id(0)
        block = tl.program_id(1)
        batch = row // num_kv_heads
        kv_head = row % num_kv_heads
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < head_dim
        key_offsets = (
            key
            + batch * key_stride_batch
            + kv_head * key_stride_head
            + offsets * key_stride_dim
        )
        value_offsets = (
            value
            + batch * value_stride_batch
            + kv_head * value_stride_head
            + offsets * value_stride_dim
        )
        key_values = tl.load(key_offsets, mask=mask)
        value_values = tl.load(value_offsets, mask=mask)
        for group_idx in tl.static_range(0, num_kv_groups):
            query_head = kv_head * num_kv_groups + group_idx
            key_cache_offsets = (
                key_cache
                + batch * key_cache_stride_batch
                + query_head * key_cache_stride_head
                + offsets * key_cache_stride_dim
            )
            value_cache_offsets = (
                value_cache
                + batch * value_cache_stride_batch
                + query_head * value_cache_stride_head
                + offsets * value_cache_stride_dim
            )
            tl.store(key_cache_offsets, key_values, mask=mask)
            tl.store(value_cache_offsets, value_values, mask=mask)

    @triton.jit
    def _store_predictor_kv_expand_gqa_cache_and_attention_first_token_kernel(
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        query_stride_batch,
        query_stride_head,
        query_stride_dim,
        key_stride_batch,
        key_stride_head,
        key_stride_dim,
        value_stride_batch,
        value_stride_head,
        value_stride_dim,
        key_cache_stride_batch,
        key_cache_stride_head,
        key_cache_stride_dim,
        value_cache_stride_batch,
        value_cache_stride_head,
        value_cache_stride_dim,
        output_stride_batch,
        output_stride_head,
        output_stride_dim,
        num_kv_heads: tl.constexpr,
        num_kv_groups: tl.constexpr,
        head_dim: tl.constexpr,
        block_size: tl.constexpr,
    ):
        """Store expanded GQA K/V and materialize one-token attention output."""

        row = tl.program_id(0)
        block = tl.program_id(1)
        batch = row // num_kv_heads
        kv_head = row % num_kv_heads
        offsets = block * block_size + tl.arange(0, block_size)
        mask = offsets < head_dim
        key_offsets = (
            key
            + batch * key_stride_batch
            + kv_head * key_stride_head
            + offsets * key_stride_dim
        )
        value_offsets = (
            value
            + batch * value_stride_batch
            + kv_head * value_stride_head
            + offsets * value_stride_dim
        )
        key_values = tl.load(key_offsets, mask=mask)
        value_values = tl.load(value_offsets, mask=mask)
        for group_idx in tl.static_range(0, num_kv_groups):
            query_head = kv_head * num_kv_groups + group_idx
            key_cache_offsets = (
                key_cache
                + batch * key_cache_stride_batch
                + query_head * key_cache_stride_head
                + offsets * key_cache_stride_dim
            )
            value_cache_offsets = (
                value_cache
                + batch * value_cache_stride_batch
                + query_head * value_cache_stride_head
                + offsets * value_cache_stride_dim
            )
            tl.store(key_cache_offsets, key_values, mask=mask)
            tl.store(value_cache_offsets, value_values, mask=mask)

            query_offsets = (
                query
                + batch * query_stride_batch
                + query_head * query_stride_head
                + offsets * query_stride_dim
            )
            query_values = tl.load(query_offsets, mask=mask)
            score = tl.sum(query_values * key_values, axis=0)
            score = score * 0.08838834764831843
            is_finite = (
                (score == score) & (score > -float("inf")) & (score < float("inf"))
            )
            invalid_probability = tl.where(score == -float("inf"), 0.0, float("nan"))
            output_values = tl.where(
                is_finite, value_values, value_values * invalid_probability
            )
            output_offsets = (
                output
                + batch * output_stride_batch
                + query_head * output_stride_head
                + offsets * output_stride_dim
            )
            tl.store(output_offsets, output_values, mask=mask)

else:
    _gather_codec_embedding_and_add_kernel = None
    _store_predictor_kv_cache_kernel = None
    _store_predictor_kv_and_expand_gqa_first_token_kernel = None
    _store_predictor_kv_and_expand_gqa_cache_kernel = None
    _store_predictor_kv_expand_gqa_cache_and_attention_first_token_kernel = None


def _contiguous_storage_ranges_overlap(
    first: torch.Tensor, second: torch.Tensor
) -> bool:
    first_start = first.data_ptr()
    first_end = first_start + first.numel() * first.element_size()
    second_start = second.data_ptr()
    second_end = second_start + second.numel() * second.element_size()
    return first_start < second_end and second_start < first_end


def _positive_stride_storage_ranges_overlap(
    first: torch.Tensor, second: torch.Tensor
) -> bool:
    """Return whether two positive-stride tensor views can address one byte."""

    if first.numel() == 0 or second.numel() == 0:
        return False
    first_start = first.data_ptr()
    second_start = second.data_ptr()
    first_end = first_start + first.element_size() * (
        1
        + sum((size - 1) * stride for size, stride in zip(first.shape, first.stride()))
    )
    second_end = second_start + second.element_size() * (
        1
        + sum(
            (size - 1) * stride for size, stride in zip(second.shape, second.stride())
        )
    )
    return first_start < second_end and second_start < first_end


def _predictor_gqa_cache_contract(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> tuple[int, int, int, int] | None:
    """Validate one-token BF16 K/V and an expanded query-head cache slot."""

    if not (
        key.is_cuda
        and value.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        return None
    if not (
        key.ndim == value.ndim == key_cache.ndim == value_cache.ndim == 4
        and key.shape == value.shape
        and key_cache.shape == value_cache.shape
    ):
        return None
    batch_size, num_kv_heads, sequence_length, head_dim = key.shape
    if batch_size == 0 or num_kv_heads == 0 or sequence_length != 1 or head_dim == 0:
        return None
    num_query_heads = key_cache.shape[1]
    if (
        key_cache.shape[0] != batch_size
        or key_cache.shape[2:] != (1, head_dim)
        or num_query_heads < num_kv_heads
        or num_query_heads % num_kv_heads
    ):
        return None
    if not (
        key.dtype
        == value.dtype
        == key_cache.dtype
        == value_cache.dtype
        == torch.bfloat16
        and key.device == value.device == key_cache.device == value_cache.device
    ):
        return None
    if any(
        stride <= 0
        for tensor in (key, value, key_cache, value_cache)
        for stride in tensor.stride()
    ):
        return None
    if any(
        _positive_stride_storage_ranges_overlap(first, second)
        for first, second in (
            (key_cache, value_cache),
            (key_cache, key),
            (key_cache, value),
            (value_cache, key),
            (value_cache, value),
        )
    ):
        return None
    return batch_size, num_kv_heads, num_query_heads, head_dim


def gather_codec_embedding_and_add(
    token_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    gathered: torch.Tensor,
    accumulated: torch.Tensor,
) -> bool:
    """Gather BF16 embedding rows and add them to an accumulator in one launch.

    Return ``False`` without writes when the caller must use the eager path.
    """

    if _gather_codec_embedding_and_add_kernel is None:
        return False
    if not (
        token_ids.is_cuda
        and embedding_weight.is_cuda
        and gathered.is_cuda
        and accumulated.is_cuda
    ):
        return False
    if token_ids.ndim != 1 or embedding_weight.ndim != 2:
        return False
    if gathered.ndim != 2 or accumulated.ndim != 2:
        return False
    batch_size = token_ids.shape[0]
    hidden_size = embedding_weight.shape[1]
    if batch_size == 0 or hidden_size == 0:
        return False
    if gathered.shape != (batch_size, hidden_size):
        return False
    if accumulated.shape != (batch_size, hidden_size):
        return False
    if token_ids.dtype not in (torch.int32, torch.int64):
        return False
    if (
        embedding_weight.dtype != torch.bfloat16
        or gathered.dtype != torch.bfloat16
        or accumulated.dtype != torch.bfloat16
    ):
        return False
    if not (
        token_ids.device
        == embedding_weight.device
        == gathered.device
        == accumulated.device
    ):
        return False
    if (
        not token_ids.is_contiguous()
        or not embedding_weight.is_contiguous()
        or not gathered.is_contiguous()
        or not accumulated.is_contiguous()
    ):
        return False
    if (
        token_ids.stride(0) != 1
        or embedding_weight.stride(1) != 1
        or gathered.stride(1) != 1
        or accumulated.stride(1) != 1
    ):
        return False
    if (
        _contiguous_storage_ranges_overlap(gathered, accumulated)
        or _contiguous_storage_ranges_overlap(gathered, embedding_weight)
        or _contiguous_storage_ranges_overlap(accumulated, embedding_weight)
    ):
        return False

    block_size = 256
    grid = (batch_size, triton.cdiv(hidden_size, block_size))
    _gather_codec_embedding_and_add_kernel[grid](
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
        token_ids.stride(0),
        embedding_weight.stride(0),
        gathered.stride(0),
        accumulated.stride(0),
        hidden_size=hidden_size,
        block_size=block_size,
        num_warps=4,
    )
    return True


def store_predictor_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> bool:
    """Store one Predictor key/value token in one capture-safe CUDA launch.

    ``key_cache`` and ``value_cache`` are slices at the target decode position.
    The operation contains only BF16 loads and stores, so it preserves the
    reference cache bits exactly. Return ``False`` without writes when its
    narrow graph-capture contract is not met.
    """

    if _store_predictor_kv_cache_kernel is None:
        return False
    if not (
        key.is_cuda
        and value.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and torch.cuda.is_current_stream_capturing()
    ):
        return False
    if not (
        key.ndim == value.ndim == key_cache.ndim == value_cache.ndim == 4
        and key.shape == value.shape == key_cache.shape == value_cache.shape
    ):
        return False
    batch_size, num_kv_heads, sequence_length, head_dim = key.shape
    if batch_size == 0 or num_kv_heads == 0 or sequence_length != 1 or head_dim == 0:
        return False
    if not (
        key.dtype
        == value.dtype
        == key_cache.dtype
        == value_cache.dtype
        == torch.bfloat16
        and key.device == value.device == key_cache.device == value_cache.device
    ):
        return False
    if any(
        stride <= 0
        for tensor in (key, value, key_cache, value_cache)
        for stride in tensor.stride()
    ):
        return False
    if any(
        _positive_stride_storage_ranges_overlap(first, second)
        for first, second in (
            (key_cache, value_cache),
            (key_cache, key),
            (key_cache, value),
            (value_cache, key),
            (value_cache, value),
        )
    ):
        return False

    block_size = 256
    grid = (batch_size * num_kv_heads, triton.cdiv(head_dim, block_size))
    _store_predictor_kv_cache_kernel[grid](
        key,
        value,
        key_cache,
        value_cache,
        key.stride(0),
        key.stride(1),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(3),
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )
    return True


def store_predictor_kv_and_expand_gqa_first_token(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    *,
    allow_eager: bool = False,
) -> bool:
    """Fuse Predictor's first-token KV write and GQA attention result.

    The Predictor cache starts at length one for the initial conditioning
    token. At that point attention's sole probability is exactly one, so every
    query head receives the corresponding KV-group value. This function writes
    the K/V cache slot and the expanded attention result in one capture-safe
    launch. It has a deliberately narrow BF16 CUDA-graph contract and returns
    ``False`` without writes outside it. ``allow_eager`` exists only for the
    graph builder's warmup launch, which compiles this Triton specialization
    before graph capture. Request decode must leave it ``False``.
    """

    if _store_predictor_kv_and_expand_gqa_first_token_kernel is None:
        return False
    if not (
        query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and key_cache.is_cuda
        and value_cache.is_cuda
        and output.is_cuda
        and (allow_eager or torch.cuda.is_current_stream_capturing())
    ):
        return False
    if not (
        query.ndim
        == key.ndim
        == value.ndim
        == key_cache.ndim
        == value_cache.ndim
        == output.ndim
        == 4
        and key.shape == value.shape == key_cache.shape == value_cache.shape
    ):
        return False
    batch_size, num_kv_heads, sequence_length, head_dim = key.shape
    if batch_size == 0 or num_kv_heads == 0 or sequence_length != 1 or head_dim == 0:
        return False
    if output.shape[0] != batch_size or output.shape[2:] != (1, head_dim):
        return False
    num_query_heads = output.shape[1]
    if num_query_heads < num_kv_heads or num_query_heads % num_kv_heads:
        return False
    if query.shape != output.shape:
        return False
    if not (
        query.dtype
        == key.dtype
        == value.dtype
        == key_cache.dtype
        == value_cache.dtype
        == output.dtype
        == torch.bfloat16
        and query.device
        == key.device
        == value.device
        == key_cache.device
        == value_cache.device
        == output.device
    ):
        return False
    if any(
        stride <= 0
        for tensor in (query, key, value, key_cache, value_cache, output)
        for stride in tensor.stride()
    ):
        return False
    if any(
        _positive_stride_storage_ranges_overlap(first, second)
        for first, second in (
            (key_cache, value_cache),
            (key_cache, query),
            (key_cache, key),
            (key_cache, value),
            (key_cache, output),
            (value_cache, key),
            (value_cache, value),
            (value_cache, output),
            (value_cache, query),
            (output, key),
            (output, value),
            (output, query),
        )
    ):
        return False

    block_size = 256
    grid = (batch_size * num_kv_heads, triton.cdiv(head_dim, block_size))
    _store_predictor_kv_and_expand_gqa_first_token_kernel[grid](
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        query.stride(0),
        query.stride(1),
        query.stride(3),
        key.stride(0),
        key.stride(1),
        key.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(3),
        num_kv_heads=num_kv_heads,
        num_kv_groups=num_query_heads // num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        num_warps=4,
    )
    return True
