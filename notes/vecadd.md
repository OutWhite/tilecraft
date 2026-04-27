# VecAdd

## Triton

Triton exposes a CTA/program model close to CUDA blocks, but the code inside a program is vectorized:

- `pid = tl.program_id(0)` selects the logical block.
- `offsets = pid * BLOCK + tl.arange(0, BLOCK)` builds a vector of element indices.
- `mask = offsets < N` handles the tail block.
- `tl.load` and `tl.store` operate on the whole vector.

For vecadd, `BLOCK_SIZE = 1024` is enough to create coalesced contiguous loads/stores. The kernel is memory bandwidth bound, so the main concerns are contiguous access, no branchy scalar loop, a clean tail predicate, and avoiding unnecessary cache pollution on streaming inputs.

## CuTe DSL

CuTe DSL keeps more of CUTLASS/CuTe's layout vocabulary visible:

- `thr_layout` describes the thread layout inside one CTA.
- `val_layout` describes how many values each thread owns.
- `make_layout_tv` combines them into a thread-value layout.
- `zipped_divide` tiles the global tensors into CTA-sized blocks.
- `partition_S` gives each thread its slice of the source/destination tile.
- fragments are register tensors used between global load and store.
- `elem_less` builds the tail predicate from the coordinate tensor.

The vecadd implementation uses 128 threads per CTA and 128-bit vectorized per-thread copy width. For `float32`, that means each thread owns 4 elements and each CTA covers 512 elements.
