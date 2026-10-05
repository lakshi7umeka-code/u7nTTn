import torch
import triton
import triton.language as tl

# =====================================================================
# 1. OPTIMIZED TRITON KERNEL
# =====================================================================

@triton.jit
def layernorm_forward_profile_kernel(
    Y_ptr, Y_stride, 
    X_ptr, X_stride, 
    W_ptr, b_ptr,          
    n_cols, eps, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_stride
    X_ptr += row_idx * X_stride

    # Pipeline Step A: Load matrix rows immediately
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
# Pipeline Step C: Latent-hidden invariant weights fetch
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Pipeline Step B: Local math calculations
    mean = tl.sum(X_row, axis=0) / n_cols
    x_centered = tl.where(mask, X_row - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    inv_std = tl.math.rsqrt(variance + eps)

    

    # Pipeline Step D: Store back out
    output = (x_centered * inv_std) * W_row + b_row
    tl.store(Y_ptr + col_offsets, output, mask=mask)


class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps=1e-5):
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim).contiguous()
        W = W.contiguous()
        b = b.contiguous()
        n_rows, n_cols = X.shape
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        Y = torch.empty_like(X)

        # Launching the kernel
        layernorm_forward_profile_kernel[(n_rows,)](
            Y, Y.stride(0), X, X.stride(0), W, b,
            n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=8
        )
        return Y.view(*shape)

def triton_layernorm(X, W, b, eps=1e-5):
    return CustomLayerNorm.apply(X, W, b, eps)


# =====================================================================
# 2. TARGETED PROFILE CONTROL SUITE
# =====================================================================

if __name__ == "__main__":
    # Force clean environment configurations
    device = "cuda"
    dtype = torch.float16
    
    # Simulating a production LLM layer shape (Batch=2, Seq=2048, Hidden=4096)
    BATCH_SIZE = 3
    SEQ_LEN = 4096
    HIDDEN_DIM = 4096 
    
    # 1. Setup Static Data allocations
    X = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=dtype, device=device)
    W = torch.ones((HIDDEN_DIM,), dtype=dtype, device=device)
    b = torch.zeros((HIDDEN_DIM,), dtype=dtype, device=device)
    eps = 1e-5

    print("⚡ Triggering Warmup Iterations (Compiling Triton IR)...")
    # Always warm up so NCU doesn't profile the Python compiler/codegen step!
    for _ in range(5):
        _ = triton_layernorm(X, W, b, eps=eps)
    torch.cuda.synchronize()

    print("🎯 Starting Target NCU Profile Action Room...")
    
    # --- Nsight Compute Targeted Context Blocks ---
    # Tells NCU to wake up right here
    torch.cuda.cudart().cudaProfilerStart() 
    
    # Emits an application range bar into Nsight Systems / Compute UI timelines
    torch.cuda.nvtx.range_push("Triton_LayerNorm_Hot_Execution")
    
    # Execute the hot path kernel we actually want to optimize
    output_tensor = triton_layernorm(X, W, b, eps=eps)
    
    # Wait for the GPU to completely finish execution before turning off the profiler
    torch.cuda.synchronize() 
    
    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop() # Tells NCU to go back to sleep
    # -----------------------------------------------

    print("🎉 Done! Profiling window completed successfully.")
