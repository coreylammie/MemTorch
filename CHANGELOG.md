## Added

1. Added Patching Support for `torch.nn.Sequential` containers.
2. Added support for modeling source and line resistances for passive crossbars/tiles.
3. Added C++ and CUDA bindings for modeling source and line resistances for passive crossbars/tiles\*.
4. Added a new MemTorch logo to `README.md`
5. Added the `set_cuda_malloc_heap_size` routine to patched `torch.mn` modules.
6. Unit test for source and line resistance modeling.

**\*Note** it is strongly suggested to set `cuda_malloc_heap_size` using `m.set_cuda_malloc_heap_size` manually when simulating source and line resisitances using CUDA bindings.

## Enhanced

1. Modularized patching logic in `memtorch.bh.nonideality.NonIdeality` and `memtorch.mn.Module`.
2. Updated `ReadTheDocs` documentation.
3. Transitioned from `Gitter` to `GitHub Discussions` for general discussion.
