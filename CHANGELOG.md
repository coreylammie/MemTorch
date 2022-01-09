## Added

1. Partial support for the `groups` argument for convolutional layers.

## Fixed

1. Patching procedure in `memtorch.mn.module.patch_model` and `memtorch.bh.nonideality.apply_nonidealities` to fix semantic error in `Tutorial.ipynb`.
2. Import statement in `Exemplar_Simulations.ipynb`.

## Enhanced

1. Further modularized patching logic in `memtorch.bh.nonideality.NonIdeality` and `memtorch.mn.Module`.
2. Modified default number of worker in `memtorch.utils` from 2 to 1.
