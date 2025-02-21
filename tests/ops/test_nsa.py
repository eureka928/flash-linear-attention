# -*- coding: utf-8 -*-

import os

import pytest
import torch
import triton

from fla.ops.common.utils import prepare_sequence_indices
from fla.ops.nsa.naive import naive_nsa
from fla.ops.nsa.parallel import parallel_nsa
from utils import assert_close


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [256, 1024, 2000])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("D", [100, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("scale", [0.1])
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    S: int,
    block_size: int,
    dtype: torch.dtype,
    scale: float
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    do = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda')

    indices = torch.full((B, T, H, S), T, dtype=torch.long, device='cuda')
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                indices[b, t, h, :len(i_i)] = i_i
    indices = indices.sort(-1)[0]

    ref = naive_nsa(q=q, k=k, v=v, indices=indices, block_size=block_size, scale=scale)
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None

    tri = parallel_nsa(q=q, k=k, v=v, indices=indices, block_size=block_size, scale=scale)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None

    assert_close(" o", ref, tri, 0.005)
    assert_close("dq", ref_dq, tri_dq, 0.005)


@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [64, 128, 200, 250, 256, 300, 400, 512, 1000, 2048])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("HQ", [64])
@pytest.mark.parametrize("D", [100, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_parallel_varlen(
    N: int,
    T: int,
    H: int,
    HQ: int,
    D: int,
    S: int,
    block_size: int,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    # randomly split the sequence into N segments
    offsets = torch.cat([
        torch.tensor([0], dtype=torch.long),
        torch.arange(16, T)[torch.randperm(T - 1)[:N-1]],
        torch.tensor([T], dtype=torch.long)
    ], 0).cuda().sort()[0]
    # seq-first required for inputs with variable lengths
    q = torch.randn((1, T, HQ, D), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((1, T, H, D), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn((1, T, HQ, D), dtype=dtype, device='cuda')

    indices = torch.full((1, T, H, S), T, dtype=torch.long, device='cuda')
    seq_indices = prepare_sequence_indices(offsets).tolist()

    for i in range(T):
        _, t = seq_indices[i]
        for h in range(H):
            i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
            indices[0, i, h, :len(i_i)] = i_i
    indices = indices.sort(-1)[0]

    ref = naive_nsa(
        q=q,
        k=k,
        v=v,
        indices=indices,
        block_size=block_size,
        cu_seqlens=offsets
    )
    ref.backward(do)
    ref_dq, q.grad = q.grad.clone(), None

    tri = parallel_nsa(
        q=q,
        k=k,
        v=v,
        indices=indices,
        block_size=block_size,
        cu_seqlens=offsets
    )
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None

    assert_close("  o", ref, tri, 0.004)
    assert_close("dq", ref_dq, tri_dq, 0.005)
