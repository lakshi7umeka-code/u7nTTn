import torch
import triton
import triton.language as tl


@triton.jit
def matmul_krnl(m,n,k,block_m:tl.constexpr,block_n:tl.constexpr,block_k:tl.constexpr,std_k,std_n,a_ptr,b_ptr,c_ptr):
    pid_m=tl.program_id(0)
    pid_n=tl.program_id(1)
    offsets_m=pid_m*block_m+tl.arange(0,block_m)
    offsets_n=pid_n*block_n+tl.arange(0,block_n)
    offsets_k=tl.arange(0,block_k)
    mask_m=offsets_m[:,None]<m
    mask_n=offsets_n[None,:]<n
    a_ptrs=a_ptr+tl.load(offsets_m[:,None]*std_k+offsets_k[None,:])
    b_ptrs=b_ptr+tl.load(offsets_k[:,None]*std_n+offsets_n[None,:])
    c_ptrs=c_ptr+tl.load(offsets_m[:,None]*std_n+offsets_n[None,:])
    tl.store(c_ptrs)

@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)

    y = tl.maximum(x, 0)

    tl.store(y_ptr + offsets, y, mask=mask)

@triton.jit
def sftmx(m,n,k,a_ptr,b_ptr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * m + tl.arange(0, m)
    offsets_n = pid_n * n + tl.arange(0, n)
    a_ptrs=a_ptr+tl.load(offsets_m[:,None]*n+offsets_n[None,:])
    row_max = tl.max(a_ptrs, axis=1)
    a_ptrs = a_ptrs - row_max[:, None]
    a_ptrs = tl.exp(a_ptrs)
    row_sum = tl.sum(a_ptrs,axis=1)
    a_ptrs=a_ptrs/row_sum[:,None]
    tl.store(b_ptr + offsets_m[:, None] * n + offsets_n[None, :], a_ptrs)
    

@triton.jit
def swiGlu(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Perform SwiGLU computation (example placeholder - replace with actual SwiGLU logic)


    y=tl.sigmoid(x)
    result = x * y
    tl.store(y_ptr + offsets, result, mask=mask)


x = torch.tensor(
    [-3, -1, 0, 2, 5, -7, 4, 8],
    device="cuda"
)

y = torch.empty_like(x)

n_elements = x.numel()

grid = (triton.cdiv(n_elements, 4),)

swiGlu[grid](
    x,
    y,
    n_elements,
    BLOCK_SIZE=4,
)

print("x:", x)
print("y:", y)