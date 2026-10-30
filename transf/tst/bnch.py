import torch
import triton
import triton.language as tl

# 1. DEFINE A BASIC TRITON KERNEL (Vector Addition)
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

def triton_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output

# 2. CONFIGURE THE PERFORMANCE REPORT
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # The variable we change on the X-axis (Tensor size)
        x_vals=[2**i for i in range(12, 24)],  # Test sizes from 4,096 to ~8.3 Million elements
        x_log=True,  # Plot the X-axis on a log scale
        line_arg='provider',  # Each line on the chart represents a different code backend
        line_vals=['triton', 'torch'],  # Label for each line
        line_names=['Triton Kernel', 'Native PyTorch'],  # Display names
        styles=[('blue', '-'), ('red', '--')],  # Visual line styles
        ylabel='Bandwidth (GB/s)',  # Metric plotted on the Y-axis
        plot_name='Vector Addition Performance (RTX 3050)',
        args={},  # Extra constant arguments passed to the benchmark
    )
)
def benchmark(size, provider):
    # Allocate tensors in VRAM on the laptop GPU
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Define execution warmups and iterations for statistical stability
    quantiles = [0.5, 0.2, 0.8]  # Median, 20th percentile, 80th percentile
    
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_add(x, y), quantiles=quantiles)
        
    # Calculate GB/s: Total Bytes Read/Written divided by Execution Time
    # float32 = 4 bytes. We Read x (size*4), Read y (size*4), and Write output (size*4)
    total_bytes = size * 4 * 3 
    gbps = total_bytes / (ms * 1e-3) / 1e9  # Convert ms to seconds and bytes to GB
    return gbps

# 3. RUN THE UTILITY
if __name__ == '__main__':
    # This executes the sweep and saves a PNG graph directly to your folder!
    benchmark.run(save_path='.', print_data=True)
