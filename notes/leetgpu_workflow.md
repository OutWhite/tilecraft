# LeetGPU Workflow

Goal: first make the submission loop reliable, then optimize for B200.

## Distinguish Failures

If the page shows:

```text
Spinning up B200...
Execution timed out
```

before compile logs or runtime logs appear, treat it as a platform provisioning/session problem first.

If the page shows compiler output, traceback, wrong answer, or measured runtime, treat it as a code problem.

Earlier `Invalid authentication token` also points to a page/session issue rather than a kernel issue.

## Minimal Chain Test

1. Refresh the problem page.
2. Log out and log in again if `Invalid authentication token` appears.
3. Submit the exact starter once:

```python
import cutlass
import cutlass.cute as cute


# A, B, C are tensors on the GPU
@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    pass
```

Expected result is wrong answer or similar, not provisioning timeout.

4. If starter also times out at `Spinning up B200`, wait and retry. The kernel has not been reached.
5. Once starter reaches a real judge result, submit `cute_dsl_kernels/vector_addition.py`.
6. Record the full output: compile error, runtime, bandwidth, wrong answer, or timeout phase.

## Optimization Loop

For every variant, record:

```text
date:
problem:
GPU:
variant:
result:
runtime_ms:
effective_GBps:
notes:
```

For vector addition at `N = 25,000,000`:

```text
effective_GBps = 300 / runtime_ms
```

where 300 MB is 2 input loads plus 1 output store.
