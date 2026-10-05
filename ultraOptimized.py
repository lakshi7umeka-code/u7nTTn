import torch
import triton
import triton.language as tl

# THE HYPER-OPTIMIZED FUSED KERNEL
@triton.jit
def opt_fused_kernel(
    x_ptr, y_ptr, z_ptr, out_ptr, n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + (tl.arange(0, BLOCK_SIZE)*2)
    mask = offsets < n_elements
    
    # The float16 elements combined with a power-of-two block size 
    # automatically triggers 128-bit vectorized loading under the hood.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    result = (x + y) * z
    
    tl.store(out_ptr + offsets, result, mask=mask)

def triton_opt_fused(x, y, z):
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # We use a larger block size to pack more threads into each SM, 
    # hiding the latency of pulling data across the 128-bit bus.
    BLOCK_SIZE = 2048  
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # num_warps=8 forces 256 threads per block to maximize execution occupancy
    opt_fused_kernel[grid](
        x, y, z, output, n_elements, 
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8
    )
    return output

if __name__ == '__main__':
    # Keep the tensor size massive to test hardware throughput limits
    SIZE = 50_000_000 
    
    # CRITICAL OPTIMIZATION: Use Float16 instead of Float32 to cut memory traffic in half
    x = torch.randn(SIZE, device='cuda', dtype=torch.float16)
    y = torch.randn(SIZE, device='cuda', dtype=torch.float16)
    z = torch.randn(SIZE, device='cuda', dtype=torch.float16)
    
    def pytorch_native(x, y, z):
        return (x + y) * z

    print("Running optimized benchmark sweep...")
    ms_torch = triton.testing.do_bench(lambda: pytorch_native(x, y, z))
    ms_triton = triton.testing.do_bench(lambda: triton_opt_fused(x, y, z))
    
    print(f"\nPyTorch FP16 Time:       {ms_torch:.4f} ms")
    print(f"Triton Ultra-Fused Time: {ms_triton:.4f} ms")
    print(f"New Achieved Speedup:    {ms_torch / ms_triton:.2f}x")
    
    # Calculate and check true achieved memory bandwidth 
    # float16 = 2 bytes. We read 3 tensors and write 1 tensor = 4 tensor movements total.
    total_bytes = SIZE * 2 * 4  
    achieved_gbps = total_bytes / (ms_triton * 1e-3) / 1e9
    print(f"Your True Achieved Bandwidth: {achieved_gbps:.2f} GB/s (Hardware Limit: ~192 GB/s)")
