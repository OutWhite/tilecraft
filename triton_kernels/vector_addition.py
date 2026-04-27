import triton
import triton.language as tl


BLOCK_SIZE = 1024
NUM_WARPS = 4


@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    a = tl.load(
        a_ptr + offsets,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )
    b = tl.load(
        b_ptr + offsets,
        mask=mask,
        other=0.0,
        cache_modifier=".cg",
        eviction_policy="evict_first",
    )

    tl.store(c_ptr + offsets, a + b, mask=mask, cache_modifier=".wb")


# A, B, C are tensors on the GPU.
def solve(A, B, C, N):
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](A, B, C, N, BLOCK=BLOCK_SIZE, num_warps=NUM_WARPS)
