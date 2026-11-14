import triton
import triton.language as tl


@triton.jit
def ffn_kernel(
    m,
    k,
    stride_am,
    stride_ak,
    attention_output_ptr,
    model_weights_ptr,
    ffn_out_ptr,
    batch,
    hidden,
    intermediate,
    sequence,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compute a SwiGLU FFN for [batch, sequence, hidden] input.
    The flat weights contain row-major matrices in this order:
    W_up [hidden, intermediate], W_gate [hidden, intermediate], and
    W_down [intermediate, hidden]. ``m`` must equal ``batch * sequence`` and
    ``k`` must equal ``hidden``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    intermediate_offsets = tl.arange(0, BLOCK_N)

    wup_base = model_weights_ptr
    wgate_base = model_weights_ptr + hidden * intermediate
    wdown_base = model_weights_ptr + 2 * hidden * intermediate

    down_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for intermediate_start in range(0, tl.cdiv(intermediate, BLOCK_N)):
        current_intermediate = (
            intermediate_start * BLOCK_N + intermediate_offsets
        )
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, tl.cdiv(k, BLOCK_K)):
            current_k = k_start * BLOCK_K + k_offsets
            x = tl.load(
                attention_output_ptr
                + row_offsets[:, None] * stride_am
                + current_k[None, :] * stride_ak,
                mask=(row_offsets[:, None] < m) & (current_k[None, :] < k),
                other=0.0,
            )
            wup = tl.load(
                wup_base
                + current_k[:, None] * intermediate
                + current_intermediate[None, :],
                mask=(current_k[:, None] < k)
                & (current_intermediate[None, :] < intermediate),
                other=0.0,
            )
            wgate = tl.load(
                wgate_base
                + current_k[:, None] * intermediate
                + current_intermediate[None, :],
                mask=(current_k[:, None] < k)
                & (current_intermediate[None, :] < intermediate),
                other=0.0,
            )
            up_acc += tl.dot(x, wup, input_precision="ieee")
            gate_acc += tl.dot(x, wgate, input_precision="ieee")

        swiglu = up_acc * tl.sigmoid(up_acc) * gate_acc
        wdown = tl.load(
            wdown_base
            + current_intermediate[:, None] * hidden
            + out_offsets[None, :],
            mask=(current_intermediate[:, None] < intermediate)
            & (out_offsets[None, :] < hidden),
            other=0.0,
        )
        down_acc += tl.dot(swiglu, wdown, input_precision="ieee")

    out_ptrs = ffn_out_ptr + row_offsets[:, None] * hidden + out_offsets[None, :]
    tl.store(
        out_ptrs,
        down_acc,
        mask=(row_offsets[:, None] < m) & (out_offsets[None, :] < hidden),
    )
