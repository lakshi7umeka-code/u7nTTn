import re
from pathlib import Path

import torch
import triton

from attn_krnl import attn_krnl


def main() -> None:
    m, n, k = 19, 35, 40
    head_dim = 10
    block_m = block_n = 16
    block_k = 32
    # Offset pointers so this PTX does not assume stronger alignment than float alignment.
    a_storage = torch.empty((m * k + 1,), device="cuda", dtype=torch.float32)
    b_storage = torch.empty((n * k + 1,), device="cuda", dtype=torch.float32)
    v_storage = torch.empty((n * k + 1,), device="cuda", dtype=torch.float32)
    c_storage = torch.empty((m * k + 1,), device="cuda", dtype=torch.float32)
    a = a_storage[1:].view(m, k)
    b = b_storage[1:].view(n, k)
    v = v_storage[1:].view(n, k)
    c = c_storage[1:].view(m, k)
    a.normal_()
    b.normal_()
    v.normal_()
    half_dim = head_dim // 2
    angles = torch.randn((max(m, n), half_dim), device="cuda", dtype=torch.float32)
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    compiled = attn_krnl.warmup(
        m,
        n,
        k,
        head_dim,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        a,
        b,
        v,
        cos,
        sin,
        c,
        block_m,
        block_n,
        block_k,
        32,
        grid=((m + block_m - 1) // block_m, triton.cdiv(k, 32)),
    )

    attn_krnl[(triton.cdiv(m, block_m), triton.cdiv(k, 32))](
        m,
        n,
        k,
        head_dim,
        a.stride(0),
        b.stride(0),
        c.stride(0),
        a,
        b,
        v,
        cos,
        sin,
        c,
        block_m,
        block_n,
        block_k,
        32,
    )
    a_heads = a.view(m, k // head_dim, head_dim)
    b_heads = b.view(n, k // head_dim, head_dim)
    cos_q = cos[:m, None, :]
    sin_q = sin[:m, None, :]
    cos_k = cos[:n, None, :]
    sin_k = sin[:n, None, :]
    a_rot = torch.cat(
        (
            a_heads[..., :half_dim] * cos_q - a_heads[..., half_dim:] * sin_q,
            a_heads[..., :half_dim] * sin_q + a_heads[..., half_dim:] * cos_q,
        ),
        dim=-1,
    ).reshape(m, k)
    b_rot = torch.cat(
        (
            b_heads[..., :half_dim] * cos_k - b_heads[..., half_dim:] * sin_k,
            b_heads[..., :half_dim] * sin_k + b_heads[..., half_dim:] * cos_k,
        ),
        dim=-1,
    ).reshape(n, k)
    expected_scores = a_rot @ b_rot.T
    expected_probabilities = torch.softmax(expected_scores, dim=-1)
    torch.testing.assert_close(
        c,
        expected_probabilities @ v,
        rtol=1e-4,
        atol=1e-4,
    )

    ptx = compiled.asm["ptx"]
    entry = re.search(r"\.visible\s+\.entry\s+attn_krnl\s*\((.*?)\)", ptx, re.DOTALL)
    if entry is None:
        raise RuntimeError("Generated PTX does not contain the attn_krnl entry point.")

    parameters = re.findall(
        r"\.param\s+(.+?)\s+attn_krnl_param_\d+\s*,?",
        entry.group(1),
    )
    if len(parameters) != 15:
        raise RuntimeError(
            f"Expected 15 PTX parameters for orch.cpp, found {len(parameters)}."
        )
    if any(".u32" not in parameter for parameter in parameters[:7]):
        raise RuntimeError("Unexpected scalar parameter ABI in generated PTX.")
    if any(".u64" not in parameter for parameter in parameters[7:]):
        raise RuntimeError("Unexpected pointer parameter ABI in generated PTX.")
    if compiled.metadata.shared != 36864:
        raise RuntimeError(
            f"Expected 36864 bytes of Triton shared memory, found {compiled.metadata.shared}."
        )
    if compiled.metadata.num_warps != 4 or compiled.metadata.warp_size != 32:
        raise RuntimeError("Unexpected Triton launch geometry for orch.cpp.")

    output_path = Path(__file__).resolve().parents[1] / "attn.ptx"
    output_path.write_text(ptx, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
