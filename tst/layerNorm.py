import torch
import triton
import triton.language as tl

# =====================================================================
# 1. TRITON KERNELS (Modify your logic here)
# =====================================================================

@triton.jit
def layernorm_forward_kernel(
    Y_ptr, Y_stride, 
    X_ptr, X_stride, 
    W_ptr, b_ptr, 
    n_cols, eps, 
    BLOCK_SIZE: tl.constexpr
):
    # Map program ID to row index
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Advance pointers to the start of this specific row
    Y_ptr += row_idx * Y_stride
    X_ptr += row_idx * X_stride

    # Load data from global memory into SRAM
    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    

    # Compute LayerNorm statistics
    mean = tl.sum(X_row, axis=0) / n_cols
    x_centered = tl.where(mask, X_row - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols
    inv_std = tl.math.rsqrt(variance + eps)
    
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    # Apply scale (W) and shift (b)
    output = (x_centered * inv_std) * W_row + b_row

    # Store output back to global memory
    tl.store(Y_ptr + col_offsets, output, mask=mask)


# =====================================================================
# 2. PYTORCH AUTOGRAD WRAPPER
# =====================================================================

class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, eps=1e-5):
        # Flatten input to 2D matrix [rows, cols]
        shape = X.shape
        dim = shape[-1]
        X = X.reshape(-1, dim).contiguous()
        W = W.contiguous()
        b = b.contiguous()
        
        n_rows, n_cols = X.shape
        
        # Calculate block size (must be a power of 2 for Triton)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        # Allocate output tensor
        Y = torch.empty_like(X)

        # Launch grid: 1D grid where each program handles exactly 1 row
        grid = (n_rows,)
        
        layernorm_forward_kernel[grid](
            Y, Y.stride(0),
            X, X.stride(0),
            W, b,
            n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4
        )
        
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY):
        # Placeholder for backward pass modifications
        # For simplicity during forward practice, returning dummy gradients
        return dY, None, None, None


# Helper function for cleaner usage
def triton_layernorm(X, W, b, eps=1e-5):
    return CustomLayerNorm.apply(X, W, b, eps)


# =====================================================================
# 3. VERIFICATION & PRACTICE RUNNER
# =====================================================================

if __name__ == "__main__":
    # Test shape details
    BATCH_SIZE = 4
    SEQ_LEN = 128
    HIDDEN_DIM = 512
    
    # Initialize inputs on GPU
    device = "cuda"
    dtype = torch.float16
    
    X = torch.randn((BATCH_SIZE, SEQ_LEN, HIDDEN_DIM), dtype=dtype, device=device)
    W = torch.ones((HIDDEN_DIM,), dtype=dtype, device=device)
    b = torch.zeros((HIDDEN_DIM,), dtype=dtype, device=device)
    eps = 1e-5

    # 1. Run PyTorch Baseline
    torch_ln = torch.nn.functional.layer_norm(X, (HIDDEN_DIM,), weight=W, bias=b, eps=eps)

    # 2. Run Your Triton Code
    triton_ln = triton_layernorm(X, W, b, eps=eps)

    # 3. Check Correctness
    is_correct = torch.allclose(torch_ln, triton_ln, atol=1e-2, rtol=1e-2)
    
    print(f"Verification status: {'PASSED ✅' if is_correct else 'FAILED ❌'}")
    if not is_correct:
        max_diff = (torch_ln - triton_ln).abs().max().item()
        print(f"Maximum discrepancy: {max_diff}")
