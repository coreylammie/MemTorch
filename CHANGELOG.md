## Added

1. Added another version of the Data Driven Model defined using `memtorch.bh.memrsitor.Data_Driven2021`.
2. Added CPU- and GPU-bound C++ bindings for `gen_tiles`.
3. Exposed `use_bindings`.
4. Added unit tests for `use_bindings`.
5. Added `exemptAssignees` tag to `scale.yml`.
6. Created `memtorch.map.Input` to encapsulate customizable input scaling methods.
7. Added the `force_scale` input argument to the default scaling method to specify whether inputs are force scaled if they do not exceed `max_input_voltage`.
8. Added CPU and GPU bindings for `tiled_inference`.

## Enhanced

1. Modularized input scaling logic for all layer types.
2. Modularized `tile_inference` for all layer types.
3. Updated ReadTheDocs documentation.

## Fixed

1. Fixed GitHub Action Workflows for external pull requests.
2. Fixed error raised by `memtorch.map.Parameter` when `p_l` is defined.
3. Fixed semantic error in `memtorch.cpp.gen_tiles`.
