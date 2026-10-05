import torch
import triton
import triton.language as tl

# 1. THE FUSED TRITON KERNEL
@triton.jit
def fused_add_mul_kernel(x_ptr, y_ptr, z_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all inputs once from the slow VRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    z = tl.load(z_ptr + offsets, mask=mask)
    
    # Math happens inside the ultra-fast Registers / SRAM
    # Both operations happen together!
    result = (x + y) * z
    
    # Store the final answer back to VRAM exactly once
    tl.store(out_ptr + offsets, result, mask=mask)

def triton_fused(x, y, z):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_add_mul_kernel[grid](x, y, z, output, n_elements, BLOCK_SIZE=1024)
    return output

# 2. THE BENCHMARKING SUITE
if __name__ == '__main__':
    # Use a large size to saturate your 192 GB/s memory highway
    SIZE = 50_000_000 
    
    # Allocate tensors on your RTX 3050 Laptop GPU
    x = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    y = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    z = torch.randn(SIZE, device='cuda', dtype=torch.float32)
    
    # Define the PyTorch baseline function
    def pytorch_native(x, y, z):
        return (x + y) * z

    # Benchmark both methods using Triton's robust timing tool
    # It automatically handles warmups and devices syncs!
    print("Running benchmark (this may take a few seconds)...")
    ms_torch = triton.testing.do_bench(lambda: pytorch_native(x, y, z))
    ms_triton = triton.testing.do_bench(lambda: triton_fused(x, y, z))
    
    print(f"\nPyTorch Native Time: {ms_torch:.4f} ms")
    print(f"Triton Fused Time:   {ms_triton:.4f} ms")
    print(f"Speedup:             {ms_torch / ms_triton:.2f}x")

