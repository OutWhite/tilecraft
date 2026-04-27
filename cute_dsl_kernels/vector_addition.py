import cutlass
import cutlass.cute as cute


tiler_mn = (128, 16)
tiler_mn_lin = (tiler_mn[0] * tiler_mn[1],)


@cute.kernel
def elementwise_add_kernel(
    gA: cute.Tensor,
    gB: cute.Tensor,
    gC: cute.Tensor,
    cC: cute.Tensor,
    shape,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    blk_coord = ((None,), bidx)
    blkA = gA[blk_coord]
    blkB = gB[blk_coord]
    blkC = gC[blk_coord]
    blkCrd = cC[blk_coord]

    coal_load = 4
    warp_size = 32
    tv_layout = cute.make_layout(
        ((warp_size, tiler_mn[0] // warp_size), (coal_load, tiler_mn[1] // coal_load)),
        stride=((coal_load, tiler_mn[1] * warp_size), (1, coal_load * warp_size)),
    )

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), gA.element_type)

    tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn_lin).get_slice(tidx)

    thrA = tiled_copy.partition_S(blkA)
    thrB = tiled_copy.partition_S(blkB)
    thrC = tiled_copy.partition_S(blkC)
    thrCrd = tiled_copy.partition_S(blkCrd)

    frgA = cute.make_fragment_like(thrA)
    frgB = cute.make_fragment_like(thrB)
    frgC = cute.make_fragment_like(thrC)

    frgPred = cute.make_fragment(thrCrd.shape, cutlass.Boolean)

    for i in range(0, cute.size(frgPred), 1):
        val = cute.elem_less(thrCrd[i], shape)
        frgPred[i] = val

    cute.copy(copy_atom, thrA, frgA, pred=frgPred)
    cute.copy(copy_atom, thrB, frgB, pred=frgPred)

    frgC.store(frgA.load() + frgB.load())

    cute.copy(copy_atom, frgC, thrC, pred=frgPred)


@cute.jit
def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
    idC = cute.make_identity_tensor(A.shape)

    gA = cute.zipped_divide(A, tiler_mn_lin)
    gB = cute.zipped_divide(B, tiler_mn_lin)
    gC = cute.zipped_divide(C, tiler_mn_lin)
    cC = cute.zipped_divide(idC, tiler=tiler_mn_lin)

    elementwise_add_kernel(gA, gB, gC, cC, (N,)).launch(
        grid=[cute.size(gA, mode=[1]), 1, 1],
        block=[tiler_mn[0], 1, 1],
    )
