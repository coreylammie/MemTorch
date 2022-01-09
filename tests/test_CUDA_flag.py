import sys

import pytest

sys.path.insert(0, "..")


def test_CUDA_flag():
    CUDA_is_false = False
    f = open("setup.py", "r")
    for line in f.readlines():
        if line.strip() == "CUDA = False":
            CUDA_is_false = True
            break

    assert CUDA_is_false
