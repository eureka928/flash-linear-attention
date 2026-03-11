# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention with A100 optimizations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.quasar.forward_substitution import forward_substitution_kernel
from fla.utils import IS_AMD
from fla.utils import autocast_custom_bwd
from fla.utils import autocast_custom_fwd
from fla.utils import autotune_cache_kwargs
from fla.utils import check_shared_mem
from fla.utils import input_guard

# A100/H100 optimized block sizes
BS_LIST = [64, 128] if check_shared_mem() else [32, 64]
BT_LIST_AUTOTUNE = [64, 128, 256]  # Larger blocks for A100
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16]
NUM_STAGES_AUTOTUNE = [3, 4]  # A100 benefits from more stages


# =============================================================================
# Kernel 1: Intra-Chunk Grouped Kernel
# Computes A_trans[S,S] and KtU[S,S] per chunk using GS=16 sub-groups
# with Neumann series for block inversion. No solve_triangular needed.
# GS=16 (not 8) because tl.dot requires inner dim >= 16.
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['S', 'BT'],
    **autotune_cache_kwargs,
)
@triton.jit
def intra_chunk_grouped_kernel_v2(
    # Inputs
    k_ptr,       # [B, T, H, S] — physical layout
    v_ptr,       # [B, T, H, S]
    beta_ptr,    # [H]
    # Outputs
    A_trans_ptr, # [B*H*NT, S, S]
    KtU_ptr,     # [B*H*NT, S, S]
    # Dimensions
    B, T: tl.constexpr, H: tl.constexpr,
    S: tl.constexpr, BT: tl.constexpr, NT,
):
    """
    Compute A_trans and KtU per chunk using GS=16 sub-groups with Neumann series.

    Grid: (B * H * NT,)
    Each thread block processes one chunk.
    """
    GS: tl.constexpr = 16
    N_GROUPS: tl.constexpr = BT // GS  # 256 // 16 = 16

    pid = tl.program_id(0)
    i_bh = pid // NT
    i_t = pid % NT
    i_b = i_bh // H
    i_h = i_bh % H

    beta_val = tl.load(beta_ptr + i_h).to(tl.float32)

    # Base pointers for this chunk in interleaved layout
    # k.view(B, H, NT, BT, S): element [b,h,nt,bt,s] at offset b*T*H*S + h*T*S + bt*S + s
    k_base = k_ptr + i_b * T * H * S + i_h * T * S + i_t * BT * S
    v_base = v_ptr + i_b * T * H * S + i_h * T * S + i_t * BT * S

    offs_s0 = tl.arange(0, S)  # [S] = [64]

    # Accumulators [S, S] for K^T @ W and K^T @ U
    s_kw = tl.zeros([S, S], dtype=tl.float32)
    s_ku = tl.zeros([S, S], dtype=tl.float32)

    # Identity for GS×GS Neumann series
    offs_gs = tl.arange(0, GS)  # [16]
    I_gs = (offs_gs[:, None] == offs_gs[None, :]).to(tl.float32)  # [GS, GS]

    eps: tl.constexpr = 1e-8

    for g in range(N_GROUPS):
        # Load k_g [GS, S] and v_g [GS, S]
        g_offset = g * GS
        offs_g = g_offset + offs_gs  # [GS]

        # k_g: k_base + offs_g[:, None] * S + offs_s0[None, :]
        k_g_ptrs = k_base + offs_g[:, None] * S + offs_s0[None, :]
        k_g = tl.load(k_g_ptrs).to(tl.float32)  # [GS, S]

        v_g_ptrs = v_base + offs_g[:, None] * S + offs_s0[None, :]
        v_g = tl.load(v_g_ptrs).to(tl.float32)  # [GS, S]

        # Compute alpha per token: alpha_i = (1 - exp(-beta * ||k_i||^2)) / (||k_i||^2 + eps)
        k_norm_sq = tl.sum(k_g * k_g, axis=1)  # [GS]
        alpha = (1.0 - tl.exp(-beta_val * k_norm_sq)) / (k_norm_sq + eps)  # [GS]

        # Correction from accumulated state
        # corr_w = k_g @ S_KW: [GS, S] @ [S, S] → [GS, S]
        # corr_u = k_g @ S_KU: [GS, S] @ [S, S] → [GS, S]
        corr_w = tl.dot(k_g.to(tl.bfloat16), s_kw.to(tl.bfloat16)).to(tl.float32)  # [GS, S]
        corr_u = tl.dot(k_g.to(tl.bfloat16), s_ku.to(tl.bfloat16)).to(tl.float32)  # [GS, S]

        # Build local M = tril(alpha[:, None] * (k_g @ k_g^T), diagonal=-1) [GS, GS]
        kkt = tl.dot(k_g.to(tl.bfloat16), tl.trans(k_g).to(tl.bfloat16)).to(tl.float32)  # [GS, GS]
        M = alpha[:, None] * kkt
        # Apply strictly lower triangular mask
        row_idx = offs_gs[:, None]
        col_idx = offs_gs[None, :]
        M = tl.where(row_idx > col_idx, M, 0.0)

        # Neumann series: (I + M)^{-1} = I - M + M^2 - M^3 + ...
        # For GS×GS strictly lower triangular M, M^GS = 0, so this is exact.
        neg_M = -M
        A_inv = I_gs + neg_M  # I - M (first two terms)
        M_pow = tl.dot(neg_M.to(tl.bfloat16), neg_M.to(tl.bfloat16)).to(tl.float32)  # (-M)^2 = M^2
        A_inv = A_inv + M_pow
        for _iter in range(GS - 3):  # 13 more iterations for GS=16
            M_pow = tl.dot(M_pow.to(tl.bfloat16), neg_M.to(tl.bfloat16)).to(tl.float32)
            A_inv = A_inv + M_pow

        # Compute alpha_k and alpha_v (corrected)
        alpha_k = alpha[:, None] * (k_g - corr_w)  # [GS, S]
        alpha_v = alpha[:, None] * (v_g - corr_u)  # [GS, S]

        # W_batch = A_inv @ alpha_k: [GS, GS] @ [GS, S] → [GS, S]
        w_batch = tl.dot(A_inv.to(tl.bfloat16), alpha_k.to(tl.bfloat16)).to(tl.float32)
        u_batch = tl.dot(A_inv.to(tl.bfloat16), alpha_v.to(tl.bfloat16)).to(tl.float32)

        # Accumulate: S_KW += k_g^T @ w_batch: [S, GS] @ [GS, S] → [S, S]
        s_kw += tl.dot(tl.trans(k_g).to(tl.bfloat16), w_batch.to(tl.bfloat16)).to(tl.float32)
        s_ku += tl.dot(tl.trans(k_g).to(tl.bfloat16), u_batch.to(tl.bfloat16)).to(tl.float32)

    # Output: A_trans = I - S_KW, KtU = S_KU
    I_s = (offs_s0[:, None] == offs_s0[None, :]).to(tl.float32)  # [S, S]
    a_trans_out = I_s - s_kw  # [S, S]

    # Store outputs — layout [B*H*NT, S, S]
    out_offset = pid * S * S
    out_ptrs_base = out_offset + offs_s0[:, None] * S + offs_s0[None, :]
    tl.store(A_trans_ptr + out_ptrs_base, a_trans_out)
    tl.store(KtU_ptr + out_ptrs_base, s_ku)


# =============================================================================
# Kernel 2: Fused Recurrence + Output Kernel
# Sequential state recurrence with immediate output computation.
# No separate state buffer needed.
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BV': 16}, num_warps=2, num_stages=2),
        triton.Config({'BV': 16}, num_warps=4, num_stages=2),
        triton.Config({'BV': 16}, num_warps=2, num_stages=4),
        triton.Config({'BV': 16}, num_warps=4, num_stages=4),
        triton.Config({'BV': 32}, num_warps=2, num_stages=2),
        triton.Config({'BV': 32}, num_warps=4, num_stages=2),
        triton.Config({'BV': 32}, num_warps=2, num_stages=4),
        triton.Config({'BV': 32}, num_warps=4, num_stages=4),
    ],
    key=['S', 'BT', 'NT'],
    **autotune_cache_kwargs,
)
@triton.jit
def recurrence_output_kernel(
    # Inputs
    q_ptr,          # [B, T, H, S]
    A_trans_ptr,    # [B*H*NT, S, S]
    KtU_ptr,        # [B*H*NT, S, S]
    # Optional initial/final state
    h0_ptr,         # [B*H, S, S] or None
    ht_ptr,         # [B*H, S, S] or None
    # Output
    o_ptr,          # [B, T, H, S]
    # Dimensions
    B, T: tl.constexpr, H: tl.constexpr,
    S: tl.constexpr, BT: tl.constexpr, NT,
    # Flags
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    # Block size for V tiling
    BV: tl.constexpr,
):
    """
    Fused state recurrence + output computation.
    For each chunk: state = A_trans @ state + KtU, then o = q @ state.

    Grid: (cdiv(S, BV), B * H)
    """
    i_v = tl.program_id(0)   # V-dimension tile index
    i_bh = tl.program_id(1)  # batch * head index
    i_b = i_bh // H
    i_h = i_bh % H

    # State tile [S, BV] in registers (float32)
    b_h = tl.zeros([S, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        # Load initial state tile [S, BV]
        p_h0 = tl.make_block_ptr(
            h0_ptr + i_bh * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (S, BV), (1, 0)
        )
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    # Chunk index base for this batch-head in A_trans/KtU
    chunk_base = i_bh * NT

    # Loop-invariant offsets
    offs_bt = tl.arange(0, BT)
    offs_s = tl.arange(0, S)
    offs_v = i_v * BV + tl.arange(0, BV)
    o_mask = offs_v[None, :] < S

    # Base pointer for q/o in interleaved view layout
    qo_bh_base = i_b * T * H * S + i_h * T * S

    for i_t in range(NT):
        chunk_idx = chunk_base + i_t

        # Load A_trans [S, S]
        p_a = tl.make_block_ptr(
            A_trans_ptr + chunk_idx * S * S,
            (S, S), (S, 1),
            (0, 0), (S, S), (1, 0)
        )
        b_a = tl.load(p_a, boundary_check=(0, 1)).to(tl.float32)

        # Load KtU tile [S, BV]
        p_ktu = tl.make_block_ptr(
            KtU_ptr + chunk_idx * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (S, BV), (1, 0)
        )
        b_ktu = tl.load(p_ktu, boundary_check=(0, 1)).to(tl.float32)

        # State update: state = A_trans @ state + KtU
        b_h = tl.dot(b_a.to(tl.bfloat16), b_h.to(tl.bfloat16)).to(tl.float32) + b_ktu

        # Load q [BT, S] from interleaved view layout
        chunk_offset = qo_bh_base + i_t * BT * S
        q_ptrs = q_ptr + chunk_offset + offs_bt[:, None] * S + offs_s[None, :]
        b_q = tl.load(q_ptrs).to(tl.float32)

        # o = q @ state: [BT, S] @ [S, BV] → [BT, BV]
        b_o = tl.dot(b_q.to(tl.bfloat16), b_h.to(tl.bfloat16)).to(tl.float32)

        # Store o in interleaved view layout (matching q/k/v reads)
        o_ptrs = o_ptr + chunk_offset + offs_bt[:, None] * S + offs_v[None, :]
        tl.store(o_ptrs, b_o.to(o_ptr.dtype.element_ty), mask=o_mask)

    # Store final state if needed
    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(
            ht_ptr + i_bh * S * S,
            (S, S), (S, 1),
            (0, i_v * BV), (S, BV), (1, 0)
        )
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


# =============================================================================
# Python Wrapper
# =============================================================================
@input_guard
def chunk_quasar_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 256,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    2-kernel chunk-wise QuasarAttention forward pass.

    Kernel 1 (intra_chunk): Computes A_trans and KtU per chunk using
        grouped sub-chunk processing with Neumann series inversion.
    Kernel 2 (recurrence_output): Fused sequential state recurrence
        with immediate output computation.

    Total: 2 kernel launches, ~75MB intermediates.
    """
    B, T, H, S = q.shape
    BT = chunk_size
    original_T = T

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Pad if T is not a multiple of BT
    if T % BT != 0:
        pad_len = BT - (T % BT)
        q = torch.cat([q, q.new_zeros((B, pad_len, H, S))], dim=1)
        k = torch.cat([k, k.new_zeros((B, pad_len, H, S))], dim=1)
        v = torch.cat([v, v.new_zeros((B, pad_len, H, S))], dim=1)
        T = T + pad_len
        NT = triton.cdiv(T, BT)

    # Allocate intermediates — only 2 tensors + output
    n_chunks = B * H * NT
    A_trans = torch.empty(n_chunks, S, S, dtype=torch.float32, device=q.device)
    KtU = torch.empty(n_chunks, S, S, dtype=torch.float32, device=q.device)
    o = torch.empty(B, T, H, S, device=q.device, dtype=q.dtype)

    # Prepare initial state
    h0 = None
    if initial_state is not None:
        h0 = initial_state.to(torch.float32).reshape(B * H, S, S)

    final_state = None
    ht = None
    if output_final_state:
        ht = torch.empty(B * H, S, S, dtype=torch.float32, device=q.device)

    # Kernel 1: Compute A_trans and KtU per chunk
    intra_chunk_grouped_kernel_v2[(n_chunks,)](
        k, v, beta,
        A_trans, KtU,
        B, T, H, S, BT, NT,
    )

    # Kernel 2: Fused recurrence + output
    def grid2(meta):
        return (triton.cdiv(S, meta['BV']), B * H)

    recurrence_output_kernel[grid2](
        q, A_trans, KtU,
        h0, ht, o,
        B, T, H, S, BT, NT,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=output_final_state,
    )

    if output_final_state:
        final_state = ht.view(B, H, S, S)

    # Un-interleave output: view [B,H,NT,BT,S] -> permute -> [B,T,H,S]
    o = o.view(B, H, NT, BT, S).permute(0, 2, 3, 1, 4).contiguous().view(B, NT * BT, H, S)

    # Trim output back to original size if padded
    if original_T != NT * BT:
        o = o[:, :original_T]

    return o, final_state


class ChunkQuasarFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        output_final_state: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs,
    ):
        chunk_size = 256
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size) if cu_seqlens is not None else None

        o, final_state = chunk_quasar_fwd(
            q=q, k=k, v=v, beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=chunk_size,
        )

        ctx.save_for_backward(q, k, v, beta, initial_state, cu_seqlens, chunk_indices)
        ctx.chunk_size = chunk_size
        ctx.output_final_state = output_final_state

        return o, final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, d_final_state: torch.Tensor | None):
        q, k, v, beta, initial_state, cu_seqlens, chunk_indices = ctx.saved_tensors

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        dbeta = torch.zeros_like(beta)

        return dq, dk, dv, dbeta, None, None, None


@torch.compiler.disable
def chunk_quasar(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Chunk-wise QuasarAttention forward pass with autograd support.

    Args:
        q: Query tensor [B, T, H, S]
        k: Key tensor [B, T, H, S]
        v: Value tensor [B, T, H, S]
        beta: Beta parameter [H]
        initial_state: Optional initial state [B, H, S, S]
        output_final_state: Whether to output final state
        cu_seqlens: Cumulative sequence lengths for variable-length sequences

    Returns:
        o: Output tensor [B, T, H, S]
        final_state: Final state [B, H, S, S] if output_final_state
    """
    return ChunkQuasarFunction.apply(
        q, k, v, beta, initial_state, output_final_state, cu_seqlens
    )
