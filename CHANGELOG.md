## Added
1. The `random_crossbar_init` argument to memtorch.bh.Crossbar. If true, this is used to initialize crossbars to random device conductances in between 1/Ron and 1/Roff.
2. `CUDA_device_idx` to `setup.py` to allow users to specify the `CUDA` device to use when installing `MemTorch` from source.
3. Implementations of CUDA accelerated passive crossbar programming routines for the 2021 Data-Driven model.
4. A BiBTeX entry for the paper which can be used to cite the corresponding OSP paper.

## Fixed
1. In the getting started tutorial, Section 4.1 was a code cell. This has since been converted to a markdown cell.
2. OOM errors encountered when modeling passive inference routines of crossbars.

## Enhanced

1. Templated quantize bindings and fixed semantic error in `memtorch.bh.nonideality.FiniteConductanceStates`.
2. The memory consumption when modeling passive inference routines.
3. The sparse factorization method used to solve sparse linear matrix systems.
4. The `naive_program` routine for crossbar programming. The maximum number of crossbar programming iterations is now configurable.


