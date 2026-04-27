# CuTe TV Layout

Reference vecadd layout:

```python
tiler_mn = (128, 16)
tiler_mn_lin = (2048,)

coal_load = 4
warp_size = 32
tv_layout = cute.make_layout(
    ((warp_size, tiler_mn[0] // warp_size), (coal_load, tiler_mn[1] // coal_load)),
    stride=((coal_load, tiler_mn[1] * warp_size), (1, coal_load * warp_size)),
)
```

This is a thread-value layout:

```text
T dimension: (32 lanes, 4 warps)
V dimension: (4 contiguous values, 4 value groups)
```

For one thread:

```text
lane = tidx % 32
warp = tidx // 32
v0   = value_id % 4
v1   = value_id // 4

offset = lane * 4 + warp * 512 + v0 + v1 * 128
```

So each CTA covers:

```text
128 threads * 16 values/thread = 2048 float32 elements
```

For a fixed warp and `v1`, all 32 lanes cover a contiguous 128-float segment:

```text
lane 0:  [base + 0,   base + 1,   base + 2,   base + 3]
lane 1:  [base + 4,   base + 5,   base + 6,   base + 7]
...
lane 31: [base + 124, base + 125, base + 126, base + 127]
```

That is the main point: every warp iteration issues coalesced contiguous accesses, while each thread gets a small contiguous vector.

Each warp repeats this four times:

```text
v1 = 0: warp-local offsets 0..127
v1 = 1: warp-local offsets 128..255
v1 = 2: warp-local offsets 256..383
v1 = 3: warp-local offsets 384..511
```

The four warps are separated by 512 elements:

```text
warp 0: offsets 0..511
warp 1: offsets 512..1023
warp 2: offsets 1024..1535
warp 3: offsets 1536..2047
```

This is close to what we would hand-write in CUDA/CUTLASS terms: 4 warps per CTA, each warp streams a 512-float contiguous chunk in four 128-float rounds.
