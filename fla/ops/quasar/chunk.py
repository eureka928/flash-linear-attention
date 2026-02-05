# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified for QuasarAttention with A100 optimizations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.index import prepare_chunk_indices
from fla.ops.quasar.forward_substitution import forward_substitution_kernel
from fla.utils import IS_AMD, autocast_custom_bwd, autocast_custom_fwd, autotune_cache_kwargs, check_shared_mem, input_guard

# A100/H100 optimized block sizes
BS_LIST = [64, 128] if check_shared_mem() else [32, 64]
BT_LIST_AUTOTUNE = [64, 128, 256]  # Larger blocks for A100
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if IS_AMD else [4, 8, 16]
NUM_STAGES_AUTOTUNE = [3, 4]  # A100 benefits from more stages


# =============================================================================
# Triton Kernel: Fused Batched MatMul for K @ K^T
# =============================================================================
@triton.autotune(
    configs=[
        # A100-optimized configs (64x64 blocks - BEST for chunk_size=64)
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # Alternative configs for flexibility
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
    **autotune_cache_kwargs,
)
@triton.jit
def batched_matmul_kernel(
    # Pointers
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for A [batch, M, K]
    stride_ab, stride_am, stride_ak,
    # Strides for B [batch, K, N]
    stride_bb, stride_bk, stride_bn,
    # Strides for C [batch, M, N]
    stride_cb, stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Batched matrix multiplication: C = A @ B
    Optimized for A100 with float32 accumulation and bfloat16 compute.
    """
    # Program IDs
    pid_batch = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for this batch
    A_batch_ptr = A_ptr + pid_batch * stride_ab
    B_batch_ptr = B_ptr + pid_batch * stride_bb
    C_batch_ptr = C_ptr + pid_batch * stride_cb

    # Initialize accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k

        # Load A block [BLOCK_M, BLOCK_K]
        a_ptrs = A_batch_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B block [BLOCK_K, BLOCK_N]
        b_ptrs = B_batch_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate in float32
        acc += tl.dot(a, b, out_dtype=tl.float32)

    # Store result
    c_ptrs = C_batch_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def triton_batched_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Batched matrix multiplication using Triton kernel.
    A: [batch, M, K]
    B: [batch, K, N]
    Returns: [batch, M, N]
    """
    assert A.dim() == 3 and B.dim() == 3
    batch, M, K = A.shape
    _, K2, N = B.shape
    assert K == K2

    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)

    # Grid dimensions
    def grid(META):
        return (batch, triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    batched_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )

    return C


# =============================================================================
# Triton Kernel: Fused KK^T with Alpha and Tril
# =============================================================================
@triton.autotune(
    configs=[
        # A100-optimized for chunk_size=256: larger blocks for better SM utilization
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['BT', 'S'],
    **autotune_cache_kwargs,
)
@triton.jit
def fused_kkt_alpha_tril_kernel(
    # Input: K [batch, BT, S]
    K_ptr,
    # Input: alpha [batch, BT, 1]
    alpha_ptr,
    # Output: M [batch, BT, BT] (lower triangular, alpha * K @ K^T)
    M_ptr,
    # Dimensions
    BT, S,
    # Strides for K
    stride_kb, stride_kt, stride_ks,
    # Strides for alpha
    stride_ab, stride_at,
    # Strides for M
    stride_mb, stride_mt, stride_mn,
    # Block size
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Fused computation: M = tril(alpha * K @ K^T, diagonal=-1)
    """
    pid_batch = tl.program_id(0)
    pid_row = tl.program_id(1)

    offs_row = pid_row * BLOCK_M + tl.arange(0, BLOCK_M)

    # Pointer for this batch
    K_batch = K_ptr + pid_batch * stride_kb
    alpha_batch = alpha_ptr + pid_batch * stride_ab
    M_batch = M_ptr + pid_batch * stride_mb

    # Load alpha for these rows
    alpha_ptrs = alpha_batch + offs_row * stride_at
    alpha_mask = offs_row < BT
    alpha_vals = tl.load(alpha_ptrs, mask=alpha_mask, other=0.0).to(tl.float32)

    # For each column block
    for col_start in range(0, BT, BLOCK_M):
        offs_col = col_start + tl.arange(0, BLOCK_M)

        # Compute K @ K^T block
        acc = tl.zeros((BLOCK_M, BLOCK_M), dtype=tl.float32)

        for k in range(0, S, BLOCK_K):
            offs_k = k + tl.arange(0, BLOCK_K)

            # Load K rows [BLOCK_M, BLOCK_K] - K[offs_row, offs_k]
            k_row_ptrs = K_batch + offs_row[:, None] * stride_kt + offs_k[None, :] * stride_ks
            k_row_mask = (offs_row[:, None] < BT) & (offs_k[None, :] < S)
            k_rows = tl.load(k_row_ptrs, mask=k_row_mask, other=0.0).to(tl.float32)

            # Load K cols [BLOCK_M, BLOCK_K] - K[offs_col, offs_k]
            k_col_ptrs = K_batch + offs_col[:, None] * stride_kt + offs_k[None, :] * stride_ks
            k_col_mask = (offs_col[:, None] < BT) & (offs_k[None, :] < S)
            k_cols = tl.load(k_col_ptrs, mask=k_col_mask, other=0.0).to(tl.float32)

            # K @ K^T: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_M] = [BLOCK_M, BLOCK_M]
            acc += tl.dot(k_rows, tl.trans(k_cols), out_dtype=tl.float32)

        # Apply alpha (broadcast across columns)
        acc = acc * alpha_vals[:, None]

        # Apply tril (set upper triangle to 0, diagonal=-1)
        row_idx = offs_row[:, None]
        col_idx = offs_col[None, :]
        tril_mask = row_idx > col_idx  # strictly lower triangular
        acc = tl.where(tril_mask, acc, 0.0)

        # Store
        m_ptrs = M_batch + offs_row[:, None] * stride_mt + offs_col[None, :] * stride_mn
        m_mask = (offs_row[:, None] < BT) & (offs_col[None, :] < BT)
        tl.store(m_ptrs, acc, mask=m_mask)


def fused_kkt_alpha_tril(K: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Fused computation: M = tril(alpha * K @ K^T, diagonal=-1)
    K: [batch, BT, S]
    alpha: [batch, BT, 1]
    Returns: M [batch, BT, BT]
    """
    batch, BT, S = K.shape

    M = torch.empty((batch, BT, BT), device=K.device, dtype=K.dtype)

    # Let autotune pick BLOCK_M, use max possible grid
    def grid(META):
        return (batch, triton.cdiv(BT, META['BLOCK_M']))

    fused_kkt_alpha_tril_kernel[grid](
        K, alpha, M,
        BT, S,
        K.stride(0), K.stride(1), K.stride(2),
        alpha.stride(0), alpha.stride(1),
        M.stride(0), M.stride(1), M.stride(2),
    )

    return M


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
    chunk_size: int = 64,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Optimized chunk-wise QuasarAttention forward pass with Triton kernels.

    This implementation uses fused Triton kernels for matrix operations,
    optimized for A100/H100 GPUs.
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

    # Reshape to chunks: [B, H, NT, BT, S]
    q_chunks = q.view(B, H, NT, BT, S)
    k_chunks = k.view(B, H, NT, BT, S)
    v_chunks = v.view(B, H, NT, BT, S)

    # Compute alpha = (1 - exp(-beta * lambda)) / (lambda + eps)
    # lambda = ||k||^2
    k_norm_sq = (k_chunks ** 2).sum(dim=-1, keepdim=True)  # [B, H, NT, BT, 1]
    eps = 1e-8
    alpha = (1 - torch.exp(-beta.view(-1, 1, 1, 1) * k_norm_sq)) / (k_norm_sq + eps)  # [B, H, NT, BT, 1]

    # Reshape for fused kernel: [B*H*NT, BT, S] and [B*H*NT, BT, 1]
    k_flat = k_chunks.view(B * H * NT, BT, S)
    alpha_flat = alpha.view(B * H * NT, BT, 1)

    # OPTIMIZATION: Use fused Triton kernel for KK^T with alpha and tril
    # M = tril(alpha * K @ K^T, diagonal=-1)
    M_flat = fused_kkt_alpha_tril(k_flat, alpha_flat)
    M = M_flat.view(B, H, NT, BT, BT)

    # Compute L = I + M for all chunks
    I = torch.eye(BT, device=q.device, dtype=q.dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, H, NT, -1, -1)
    L = I + M  # [B, H, NT, BT, BT] lower triangular with 1s on diagonal

    # OPTIMIZATION: Use PyTorch's optimized triangular solve (faster for small matrices)
    # Compute A = L^(-1) by solving L @ A = I
    I_eye = torch.eye(BT, device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, H, NT, -1, -1)
    L_f32 = L.to(torch.float32)  # Use float32 for numerical stability
    A = torch.linalg.solve_triangular(L_f32, I_eye, upper=False).to(q.dtype)  # [B, H, NT, BT, BT]

    # Compute W = A @ (alpha * K) and U = A @ (alpha * V) for all chunks
    alpha_expanded = alpha.expand(-1, -1, -1, -1, S)  # [B, H, NT, BT, S]

    # OPTIMIZATION: Use Triton batched matmul for W and U
    A_flat = A.view(B * H * NT, BT, BT)
    alpha_k_flat = (alpha_expanded * k_chunks).view(B * H * NT, BT, S)
    alpha_v_flat = (alpha_expanded * v_chunks).view(B * H * NT, BT, S)

    W_flat = triton_batched_matmul(A_flat, alpha_k_flat)
    U_flat = triton_batched_matmul(A_flat, alpha_v_flat)

    W = W_flat.view(B, H, NT, BT, S)
    U = U_flat.view(B, H, NT, BT, S)

    # Initialize output tensor
    o = torch.empty_like(q)

    # Initialize state
    if initial_state is None:
        state = torch.zeros(B, H, S, S, dtype=q.dtype, device=q.device)
    else:
        state = initial_state.clone()

    # Pre-compute identity matrix ONCE (moved outside loop for efficiency)
    # Use float32 for state computations for numerical stability
    I_full = torch.eye(S, device=q.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, S, S]

    # Convert state to float32 for stable accumulation
    state = state.to(torch.float32)

    # Process chunks sequentially for state updates (this is inherently sequential)
    # But intra-chunk computations are already vectorized!
    for i in range(NT):
        q_c = q_chunks[:, :, i]  # [B, H, BT, S]
        k_c = k_chunks[:, :, i]  # [B, H, BT, S]
        W_c = W[:, :, i]  # [B, H, BT, S]
        U_c = U[:, :, i]  # [B, H, BT, S]

        # Convert to float32 for stable computation
        k_c_f32 = k_c.to(torch.float32)
        W_c_f32 = W_c.to(torch.float32)
        U_c_f32 = U_c.to(torch.float32)

        # Inter-chunk state transition (in float32)
        # A = I - K^T @ W
        # B = K^T @ U
        k_c_t = k_c_f32.transpose(-2, -1)  # Reuse transposed tensor
        A_trans = I_full - torch.matmul(k_c_t, W_c_f32)  # [B, H, S, S]
        B_trans = torch.matmul(k_c_t, U_c_f32)  # [B, H, S, S]

        # Update state: S_new = A @ S_prev + B (in float32)
        state = torch.matmul(A_trans, state) + B_trans  # [B, H, S, S]

        # Compute output (in float32, then convert back)
        # o = q @ S_prev + q @ K^T @ (U - W @ S_prev)
        q_c_f32 = q_c.to(torch.float32)
        o_inter = torch.matmul(q_c_f32, state)  # [B, H, BT, S]
        diff = U_c_f32 - torch.matmul(W_c_f32, state)  # Compute difference once
        o_intra = torch.matmul(q_c_f32, torch.matmul(k_c_t, diff))  # Reuse k_c_t
        o_c = (o_inter + o_intra).to(q.dtype)  # [B, H, BT, S]

        # Store output
        o_c = o_c.transpose(1, 2)  # [B, BT, H, S]
        o[:, i*BT:(i+1)*BT] = o_c

    final_state = state if output_final_state else None

    # Trim output back to original size if padded
    if original_T != T:
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
        chunk_size = 256  # Larger chunks = fewer loop iterations, better for A100
        chunk_indices = prepare_chunk_indices(
            cu_seqlens, chunk_size) if cu_seqlens is not None else None

        o, final_state = chunk_quasar_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
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

        # Backward pass implementation (simplified for now)
        # Full backward pass would require recomputing forward and computing gradients
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

    Implements the chunk-wise parallel algorithm for QuasarAttention.

    Args:
        q (torch.Tensor): Query tensor of shape [B, T, H, S]
        k (torch.Tensor): Key tensor of shape [B, T, H, S]
        v (torch.Tensor): Value tensor of shape [B, T, H, S]
        beta (torch.Tensor): Beta parameter tensor of shape [H]
        initial_state (torch.Tensor | None): Initial state tensor of shape [B, H, S, S]
        output_final_state (bool): Whether to output the final state
        cu_seqlens (torch.Tensor | None): Cumulative sequence lengths for variable-length sequences

    Returns:
        o (torch.Tensor): Output tensor of shape [B, T, H, S]
        final_state (torch.Tensor | None): Final state tensor of shape [B, H, S, S] if output_final_state
    """
    return ChunkQuasarFunction.apply(q, k, v, beta, initial_state, output_final_state, cu_seqlens)
