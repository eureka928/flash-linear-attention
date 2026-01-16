# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Token-parallel implementation of QuasarAttention

import torch
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.quasar.forward_substitution import forward_substitution_kernel
from fla.utils import IS_AMD, autotune_cache_kwargs, check_shared_mem, input_guard

NUM_WARPS = [2, 4, 8, 16] if IS_AMD else [4, 8, 16, 32]


@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'S', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['T'])
def chunk_quasar_fwd_kernel_intra(
    q,
    k,
    v,
    beta,
    KK_t,
    M,
    L,
    A,
    W,
    U,
    cu_seqlens,
    T,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Intra-chunk computation kernel - processes all chunks in parallel.
    Each chunk gets its own thread block.
    """
    i_c, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_b).to(tl.int32), tl.load(cu_seqlens + i_b + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    
    if i_c * BT >= T:
        return
    
    # Load k, v for this chunk
    # k shape: [B, T, H, S] -> need to load [BT, S] for this chunk
    k_offset = (bos * H + i_h) * S + i_c * BT * H * S
    v_offset = (bos * H + i_h) * S + i_c * BT * H * S
    beta_offset = bos * H + i_h
    
    # Load k, v, beta for this chunk
    # [BT, S]
    p_k = tl.make_block_ptr(
        base=k + k_offset,
        shape=(BT, S),
        strides=(H*S, 1),
        order=(0, 1),
        block_shape=(BT, BK),
    )
    p_v = tl.make_block_ptr(
        base=v + v_offset,
        shape=(BT, S),
        strides=(H*S, 1),
        order=(0, 1),
        block_shape=(BT, BK),
    )
    p_beta = tl.make_block_ptr(
        base=beta + beta_offset,
        shape=(BT,),
        strides=(H,),
        order=(0,),
        block_shape=(BT,),
    )
    
    b_k = tl.load(p_k, boundary_check=(0, 1)).to(tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1)).to(tl.float32)
    b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
    
    # Compute alpha = (1 - exp(-beta * lambda)) / (lambda + eps)
    # lambda = ||k||^2
    b_lambda = tl.sum(b_k * b_k, axis=1)
    eps = 1e-8
    b_alpha = (1 - tl.exp(-b_beta * b_lambda)) / (b_lambda + eps)
    
    # Compute KK^T = K @ K^T
    # [BT, S] @ [S, BT] -> [BT, BT]
    b_KK_t = tl.dot(b_k, b_k, trans_b=True)
    
    # Compute M = tril(alpha * KK^T)
    b_alpha_expanded = b_alpha[:, None]
    b_M = b_alpha_expanded * b_KK_t
    # Zero out upper triangular part (including diagonal)
    for i in range(BT):
        for j in range(i, BT):
            b_M[i, j] = 0.0
    
    # Compute L = I + M
    b_L = b_M
    for i in range(BT):
        b_L[i, i] = 1.0
    
    # Compute A = L^(-1) using forward substitution
    # This is done in a separate kernel
    
    # Compute W = A @ (alpha * K)
    # U = A @ (alpha * V)
    b_alpha_k = b_alpha_expanded * b_k
    b_alpha_v = b_alpha_expanded * b_v
    
    # Store intermediate results
    KK_t_offset = i_bh * BT * BT * T + i_c * BT * BT
    M_offset = i_bh * BT * BT * T + i_c * BT * BT
    L_offset = i_bh * BT * BT * T + i_c * BT * BT
    
    p_KK_t = tl.make_block_ptr(
        base=KK_t + KK_t_offset,
        shape=(BT, BT),
        strides=(BT, 1),
        order=(0, 1),
        block_shape=(BT, BT),
    )
    p_M = tl.make_block_ptr(
        base=M + M_offset,
        shape=(BT, BT),
        strides=(BT, 1),
        order=(0, 1),
        block_shape=(BT, BT),
    )
    p_L = tl.make_block_ptr(
        base=L + L_offset,
        shape=(BT, BT),
        strides=(BT, 1),
        order=(0, 1),
        block_shape=(BT, BT),
    )
    
    tl.store(p_KK_t, b_KK_t)
    tl.store(p_M, b_M)
    tl.store(p_L, b_L)
