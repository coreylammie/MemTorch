import pytest
import sys
sys.path.insert(0,'..')


def test_CUDA_flag():
    CUDA_is_false = False
    f = open("setup.py", "r")
    for line in f.readlines():
        if line.strip() == 'CUDA = False':
            print('wat')
            CUDA_is_false = True
            break

    assert CUDA_is_false
