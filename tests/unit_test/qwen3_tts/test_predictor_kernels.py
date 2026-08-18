# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for optional Qwen3-TTS predictor kernels."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sglang_omni.models.qwen3_tts.predictor_kernels import (
    gather_codec_embedding_and_add,
    store_predictor_kv_and_expand_gqa_first_token,
    store_predictor_kv_cache,
)


_HAS_CUDA = torch.cuda.is_available()


def test_gather_codec_embedding_and_add_cpu_falls_back_without_writes():
    token_ids = torch.tensor([1, 3], dtype=torch.long)
    embedding_weight = torch.randn(8, 4, dtype=torch.bfloat16)
    gathered = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    accumulated = torch.full((2, 4), -3.0, dtype=torch.bfloat16)
    expected_gathered = gathered.clone()
    expected_accumulated = accumulated.clone()

    assert not gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
    )
    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)


def test_store_predictor_kv_cache_cpu_falls_back_without_writes():
    key = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_cache = torch.full((2, 1, 4, 8), 2.0, dtype=torch.bfloat16)
    value_cache = torch.full((2, 1, 4, 8), -3.0, dtype=torch.bfloat16)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()

    assert not store_predictor_kv_cache(
        key,
        value,
        key_cache[:, :, 2:3, :],
        value_cache[:, :, 2:3, :],
    )
    assert torch.equal(key_cache, expected_key_cache)
    assert torch.equal(value_cache, expected_value_cache)


def test_store_predictor_kv_and_expand_gqa_first_token_cpu_falls_back_without_writes():
    query = torch.randn(2, 2, 1, 8, dtype=torch.bfloat16)
    key = torch.randn(2, 1, 1, 8, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    key_cache = torch.full((2, 1, 1, 8), 2.0, dtype=torch.bfloat16)
    value_cache = torch.full((2, 1, 1, 8), -3.0, dtype=torch.bfloat16)
    output = torch.full((2, 2, 1, 8), 5.0, dtype=torch.bfloat16)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    expected_output = output.clone()

    assert not store_predictor_kv_and_expand_gqa_first_token(
        query, key, value, key_cache, value_cache, output
    )
    assert torch.equal(key_cache, expected_key_cache)
    assert torch.equal(value_cache, expected_value_cache)
    assert torch.equal(output, expected_output)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton Predictor kernel needs CUDA")
@pytest.mark.parametrize("batch_size,num_kv_heads,head_dim", [(1, 1, 8), (4, 8, 128)])
def test_store_predictor_kv_cache_matches_reference_during_graph_capture(
    batch_size: int,
    num_kv_heads: int,
    head_dim: int,
):
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(
        batch_size * 10000 + num_kv_heads * 100 + head_dim
    )
    key = torch.randn(
        batch_size,
        num_kv_heads,
        1,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    value = torch.randn(
        batch_size,
        num_kv_heads,
        1,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    key_cache = torch.randn(
        batch_size,
        num_kv_heads,
        5,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    value_cache = torch.randn_like(key_cache)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    cache_position = 3
    expected_key_cache[:, :, cache_position : cache_position + 1, :].copy_(key)
    expected_value_cache[:, :, cache_position : cache_position + 1, :].copy_(value)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert store_predictor_kv_cache(
            key,
            value,
            key_cache[:, :, cache_position : cache_position + 1, :],
            value_cache[:, :, cache_position : cache_position + 1, :],
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(key_cache, expected_key_cache)
    assert torch.equal(value_cache, expected_value_cache)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton Predictor kernel needs CUDA")
@pytest.mark.parametrize(
    ("batch_size", "num_kv_heads", "num_query_heads", "head_dim"),
    [(1, 1, 1, 8), (4, 8, 16, 128), (8, 8, 16, 128)],
)
def test_store_predictor_kv_and_expand_gqa_first_token_matches_sdpa_during_graph_capture(
    batch_size: int,
    num_kv_heads: int,
    num_query_heads: int,
    head_dim: int,
):
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(
        batch_size * 100000 + num_kv_heads * 1000 + num_query_heads * 10 + head_dim
    )
    query = torch.randn(
        batch_size,
        num_query_heads,
        1,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    key = torch.randn(
        batch_size,
        num_kv_heads,
        1,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    value = torch.randn(
        batch_size,
        num_kv_heads,
        1,
        head_dim,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    key_cache = torch.randn_like(key)
    value_cache = torch.randn_like(value)
    expected_key_cache = key.clone()
    expected_value_cache = value.clone()
    expected_output = F.scaled_dot_product_attention(
        query,
        expected_key_cache,
        expected_value_cache,
        is_causal=False,
        enable_gqa=num_query_heads != num_kv_heads,
    )
    output = torch.empty_like(query)

    # Triton compilation is not valid inside CUDA Graph capture. Predictor graph
    # construction runs this same eager warmup before capturing the decode path.
    assert store_predictor_kv_and_expand_gqa_first_token(
        query,
        key,
        value,
        key_cache,
        value_cache,
        output,
        allow_eager=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert store_predictor_kv_and_expand_gqa_first_token(
            query,
            key,
            value,
            key_cache,
            value_cache,
            output,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(key_cache, expected_key_cache)
    assert torch.equal(value_cache, expected_value_cache)
    assert torch.equal(output, expected_output)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton Predictor kernel needs CUDA")
@pytest.mark.parametrize(
    ("query_value", "key_value"),
    [
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (float("-inf"), 1.0),
        (1.0, float("nan")),
    ],
)
def test_store_predictor_kv_and_expand_gqa_first_token_matches_sdpa_nonfinite_scores(
    query_value: float,
    key_value: float,
):
    """A one-element softmax still has defined non-finite edge behavior."""

    device = torch.device("cuda")
    query = torch.full((1, 2, 1, 8), query_value, dtype=torch.bfloat16, device=device)
    key = torch.full((1, 1, 1, 8), key_value, dtype=torch.bfloat16, device=device)
    value = torch.tensor(
        [[[[1.0, 2.0, float("nan"), float("inf"), 3.0, 4.0, 5.0, 6.0]]]],
        dtype=torch.bfloat16,
        device=device,
    )
    expected_output = F.scaled_dot_product_attention(
        query, key, value, is_causal=False, enable_gqa=True
    )
    key_cache = torch.zeros_like(key)
    value_cache = torch.zeros_like(value)
    output = torch.empty_like(query)

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert store_predictor_kv_and_expand_gqa_first_token(
            query,
            key,
            value,
            key_cache,
            value_cache,
            output,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton Predictor kernel needs CUDA")
@pytest.mark.parametrize("partial_overlap", [False, True])
def test_store_predictor_kv_cache_rejects_aliased_outputs_without_writes(
    partial_overlap: bool,
):
    device = torch.device("cuda")
    key = torch.full((1, 1, 1, 8), 2.0, dtype=torch.bfloat16, device=device)
    value = torch.full((1, 1, 1, 8), -3.0, dtype=torch.bfloat16, device=device)
    shared_cache = torch.full((1, 1, 1, 16), 7.0, dtype=torch.bfloat16, device=device)
    expected_cache = shared_cache.clone()
    key_cache = shared_cache[..., :8]
    value_cache = shared_cache[..., 4:12] if partial_overlap else key_cache

    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        assert not store_predictor_kv_cache(
            key,
            value,
            key_cache,
            value_cache,
        )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(shared_cache, expected_cache)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton predictor kernel needs CUDA")
@pytest.mark.parametrize("invalid_input", ["dtype", "layout", "overlap"])
def test_gather_codec_embedding_and_add_rejects_unsafe_input_without_writes(
    invalid_input: str,
):
    device = torch.device("cuda")
    token_ids = torch.tensor([1, 3], dtype=torch.long, device=device)
    embedding_weight = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
    accumulated = torch.full((2, 8), -3.0, dtype=torch.bfloat16, device=device)
    gathered = torch.full((2, 8), 2.0, dtype=torch.bfloat16, device=device)

    if invalid_input == "dtype":
        embedding_weight = embedding_weight.float()
    elif invalid_input == "layout":
        gathered = torch.full((8, 2), 2.0, dtype=torch.bfloat16, device=device).t()
    else:
        shared = torch.full((3, 8), 2.0, dtype=torch.bfloat16, device=device)
        gathered = shared[:2]
        accumulated = shared[1:]

    expected_gathered = gathered.clone()
    expected_accumulated = accumulated.clone()
    assert not gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
    )
    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)


@pytest.mark.skipif(not _HAS_CUDA, reason="Triton predictor kernel needs CUDA")
@pytest.mark.parametrize(
    ("batch_size", "hidden_size"),
    [(1, 8), (4, 8), (8, 2048)],
)
def test_gather_codec_embedding_and_add_matches_bf16_reference(
    batch_size: int,
    hidden_size: int,
):
    device = torch.device("cuda")
    generator = torch.Generator(device="cpu").manual_seed(
        batch_size * 10000 + hidden_size
    )
    vocab_size = 53
    token_ids = torch.randint(
        0,
        vocab_size,
        (batch_size,),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    if batch_size > 1:
        token_ids[1] = token_ids[0]
    embedding_weight = torch.randn(
        vocab_size,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    accumulated = torch.randn(
        batch_size,
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    expected_gathered = F.embedding(token_ids, embedding_weight)
    expected_accumulated = accumulated.clone()
    expected_accumulated.add_(expected_gathered)
    gathered = torch.empty_like(expected_gathered)

    assert gather_codec_embedding_and_add(
        token_ids,
        embedding_weight,
        gathered,
        accumulated,
    )
    torch.cuda.synchronize()

    assert torch.equal(gathered, expected_gathered)
    assert torch.equal(accumulated, expected_accumulated)
