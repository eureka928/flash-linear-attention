# -*- coding: utf-8 -*-

import os
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from fla.ops.generalized_delta_rule.dplr.chunk import chunk_dplr_delta_rule
from utils import assert_close

def chunk_dplr_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    scale: float = None,
    chunk_size: int = 64,
    head_first: bool = True,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        b = b.transpose(1, 2)
        gk = gk.transpose(1, 2)
    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT

    q, k, v, a, b, gk = map(lambda x: F.pad(x, (0, 0, 0, pad_len)).to(torch.float32), [q, k, v, a, b, gk])
    B, H, L, DK = q.shape
    DV = v.shape[-1]
    q = q * scale

    S = k.new_zeros(B, H, DK, DV)
    if initial_state is not None:
        S += initial_state

    # note that diagonal is masked.
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    q, k, v, a, b, gk = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, a, b, gk])
    gk_cumsum = gk.cumsum(-2)
    A_ab = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)
    A_qk = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)
    A_ak = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)
    A_qb = torch.zeros(B, H, L // chunk_size, chunk_size, chunk_size).to(q.device)

    for i in range(chunk_size):
        a_i = a[:, :, :, i, None]
        q_i = q[:, :, :, i, None]
        gk_i = gk_cumsum[:, :, :, i, None]
        mask = (torch.arange(chunk_size) <= i).to(q.device)
        attn_i = (gk_i - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_qk[:, :, :, i, :] = (q_i * k * attn_i).sum(-1).clone()
        A_qb[:, :, :, i, :] = (q_i * b * attn_i).sum(-1).clone()
        mask = (torch.arange(chunk_size) < i).to(q.device)
        # shift by one.
        attn_i = (gk_i - gk[:,:,:,i,None] - gk_cumsum).masked_fill(~mask.unsqueeze(-1), float('-inf')).exp()
        A_ab[:, :, :, i, :] = (a_i * b * attn_i).sum(-1).clone()
        A_ak[:, :, :, i, :] = (a_i * k * attn_i).sum(-1).clone()

    A_ab = A_ab
    for i in range(1, chunk_size):
        A_ab[..., i, :i] = A_ab[..., i, :i].clone() + (A_ab[..., i, :, None].clone() * A_ab[..., :, :i].clone()).sum(-2)

    A_ab = A_ab + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    u = A_ab @ (A_ak @ v)
    w = A_ab @ ((gk_cumsum-gk).exp() * a)

    o = torch.zeros_like(v)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(0, L // chunk_size):
        q_i, k_i, v_i, u_i, w_i, b_i = q[:, :, i], k[:, :, i], v[:, :, i], u[:, :, i], w[:, :, i], b[:, :, i]
        v2_i = u_i + w_i @ S
        o_1 = A_qk[:, :, i] @ v_i
        o_2 = A_qb[:, :, i] @ v2_i
        o_3 = (q_i * gk_cumsum[:, :, i].exp()) @ S
        o[:, :, i] = o_1 + o_2 + o_3
        decay = (gk_cumsum[:, :, i, -1, None] - gk_cumsum[:, :, i]).exp()
        S = S*gk_cumsum[:, :, i, -1, :, None].exp() + (k_i * decay).transpose(-1, -2) @ v_i + (b_i * decay).transpose(-1, -2) @ v2_i

    S = None if output_final_state is False else S
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    if not head_first:
        o = o.transpose(1, 2)
    return o, S


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("gate_logit_normalizer", [1, 0.1, 10])
@pytest.mark.parametrize("T", [256, 300])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("D", [256, 100])
@pytest.mark.parametrize("scale", [0.25])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    gate_logit_normalizer: float,
    dtype: torch.dtype,
    head_first: bool,
):
    if head_first:
        q = torch.randn(B, H, T, D, dtype=dtype)
        k = torch.randn(B, H, T, D, dtype=dtype)
        v = torch.randn(B, H, T, D, dtype=dtype)
        a = torch.rand(B, H, T, D, dtype=dtype)
        gk = (torch.randn(B, H, T, D, dtype=torch.float)) 
    else:
        q = torch.randn(B, T, H, D, dtype=dtype)
        k = torch.randn(B, T, H, D, dtype=dtype)
        v = torch.randn(B, T, H, D, dtype=dtype)
        a = torch.rand(B, T, H, D, dtype=dtype)
        gk = torch.randn(B, T, H, D, dtype=torch.float)

    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = -a
    gk = torch.nn.functional.logsigmoid(gk) / gate_logit_normalizer
 
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, a, b, gk, h0 = map(lambda x: x.cuda().requires_grad_(False), (q, k, v, a, b, gk, h0))
    ref, ref_ht = chunk_dplr_delta_rule_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        head_first=head_first
    )
    tri, tri_ht = chunk_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
        head_first=head_first
    )
    assert_close("  o", ref, tri, 0.007)
    assert_close(" ht", ref_ht, tri_ht, 0.008)



@pytest.mark.parametrize("N", [4])
@pytest.mark.parametrize("T", [64, 128, 200, 250, 256, 300, 400, 512, 1000, 2048])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("D", [50, 100, 200])
@pytest.mark.parametrize("scale", [0.25])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_chunk_varlen(
    N: int,
    T: int,
    H: int,
    D: int,
    scale: float,
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
    q = torch.randn(1, T, H, D, dtype=dtype)
    k = torch.randn(1, T, H, D, dtype=dtype)
    v = torch.randn(1, T, H, D, dtype=dtype)
    a = torch.rand(1, T, H, D, dtype=dtype)
    gk = torch.randn(1, T, H, D, dtype=torch.float)
    a = torch.nn.functional.normalize(a, p=2, dim=-1)
    b = -a
    gk = torch.nn.functional.logsigmoid(gk)
    h0 = torch.randn(N, H, D, D, dtype=torch.float32)
    q, k, v, a, b, gk, h0 = map(lambda x: x.cuda().requires_grad_(False), (q, k, v, a, b, gk, h0))

    tri, tri_ht = chunk_dplr_delta_rule(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        a=a.clone(),
        b=b.clone(),
        gk=gk.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        offsets=offsets,
        head_first=False
    )

    ref = []
    ref_ht = []
    for i in range(N):
        ref_i, ref_ht_i = chunk_dplr_delta_rule_ref(
            q=q[:, offsets[i]:offsets[i+1]],
            k=k[:, offsets[i]:offsets[i+1]],
            v=v[:, offsets[i]:offsets[i+1]],
            a=a[:, offsets[i]:offsets[i+1]],
            b=b[:, offsets[i]:offsets[i+1]],
            gk=gk[:, offsets[i]:offsets[i+1]],
            scale=scale,
            initial_state=h0[i],
            output_final_state=True,
            head_first=False
        )
        try:
            assert_close(f"  o_{i}", ref_i, tri[:, offsets[i]:offsets[i+1]], 0.005)
            # assert_close(f" ht_{i}", ref_ht_i, tri_ht[i], 0.005)
        except Exception as e:
            breakpoint()
            raise e



