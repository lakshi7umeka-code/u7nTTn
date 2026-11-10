import triton
import triton.language as tl


@triton.jit
def _attention_scores(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    rows,
    cols,
    m,
    n,
    k,
    head_dim,
    stride_am,
    stride_bn,
    block_size_k: tl.constexpr,
):
    k_offsets = tl.arange(0, block_size_k)
    half_dim = head_dim // 2
    acc = tl.zeros((rows.shape[0], cols.shape[0]), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(k, block_size_k)):
        current_k = k_start * block_size_k + k_offsets
        dim = current_k % head_dim
        pair_k = current_k + tl.where(dim < half_dim, half_dim, -half_dim)
        rope_dim = dim % half_dim
        k_mask = current_k[None, :] < k

        q_base = q_ptr + rows[:, None] * stride_am
        q = tl.load(
            q_base + current_k[None, :],
            mask=(rows[:, None] < m) & k_mask,
            other=0.0,
        )
        q_pair = tl.load(
            q_base + pair_k[None, :],
            mask=(rows[:, None] < m) & k_mask,
            other=0.0,
        )
        q_cos = tl.load(
            cos_ptr + rows[:, None] * half_dim + rope_dim[None, :],
            mask=(rows[:, None] < m) & k_mask,
            other=0.0,
        )
        q_sin = tl.load(
            sin_ptr + rows[:, None] * half_dim + rope_dim[None, :],
            mask=(rows[:, None] < m) & k_mask,
            other=0.0,
        )
        q_rot = tl.where(
            dim[None, :] < half_dim,
            q * q_cos - q_pair * q_sin,
            q_pair * q_sin + q * q_cos,
        )

        key_base = k_ptr + cols[:, None] * stride_bn
        key = tl.load(
            key_base + current_k[None, :],
            mask=(cols[:, None] < n) & k_mask,
            other=0.0,
        )
        key_pair = tl.load(
            key_base + pair_k[None, :],
            mask=(cols[:, None] < n) & k_mask,
            other=0.0,
        )
        k_cos = tl.load(
            cos_ptr + cols[:, None] * half_dim + rope_dim[None, :],
            mask=(cols[:, None] < n) & k_mask,
            other=0.0,
        )
        k_sin = tl.load(
            sin_ptr + cols[:, None] * half_dim + rope_dim[None, :],
            mask=(cols[:, None] < n) & k_mask,
            other=0.0,
        )
        k_rot = tl.where(
            dim[None, :] < half_dim,
            key * k_cos - key_pair * k_sin,
            key_pair * k_sin + key * k_cos,
        )
        acc += tl.dot(q_rot, tl.trans(k_rot), input_precision="ieee")

    return acc


@triton.jit
def attn_krnl(
    m,
    n,
    k,
    head_dim,
    stride_am,
    stride_bn,
    stride_cm,
    q_ptr,
    k_ptr,
    v_ptr,
    cos_ptr,
    sin_ptr,
    c_ptr,
    block_size_m: tl.constexpr,
    block_size_n: tl.constexpr,
    block_size_k: tl.constexpr,
    block_size_v: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    rows = pid_m * block_size_m + tl.arange(0, block_size_m)
    value_cols = pid_v * block_size_v + tl.arange(0, block_size_v)
    row_max = tl.full((block_size_m,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((block_size_m,), dtype=tl.float32)
    output_acc = tl.zeros((block_size_m, block_size_v), dtype=tl.float32)

    for n_start in range(0, tl.cdiv(n, block_size_n)):
        cols = n_start * block_size_n + tl.arange(0, block_size_n)
        scores = _attention_scores(
            q_ptr,
            k_ptr,
            cos_ptr,
            sin_ptr,
            rows,
            cols,
            m,
            n,
            k,
            head_dim,
            stride_am,
            stride_bn,
            block_size_k,
        )
        scores = tl.where(cols[None, :] < n, scores, float("-inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        alpha = tl.exp(row_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        values = tl.load(
            v_ptr + cols[:, None] * stride_bn + value_cols[None, :],
            mask=(cols[:, None] < n) & (value_cols[None, :] < k),
            other=0.0,
        )
        output_acc = output_acc * alpha[:, None] + tl.dot(
            probabilities, values, input_precision="ieee"
        )
        row_sum = (
            row_sum * alpha
            + tl.sum(probabilities, axis=1)
        )
        row_max = new_max

    output = output_acc / row_sum[:, None]
    output_ptrs = c_ptr + rows[:, None] * stride_cm + value_cols[None, :]
    tl.store(
        output_ptrs,
        output,
        mask=(rows[:, None] < m) & (value_cols[None, :] < k),
    )
