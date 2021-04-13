import pytest
import torch

import memtorch
from memtorch.utils import LoadCIFAR10, LoadMNIST


@pytest.mark.parametrize("dataloader", [LoadMNIST, LoadCIFAR10])
def test_dataloader(dataloader):
    train_loader, validation_loader, test_loader = dataloader(validation=True)
    assert train_loader is not None
    assert validation_loader is not None
    assert test_loader is not None
    train_loader, validation_loader, test_loader = dataloader(validation=False)
    assert train_loader is not None
    assert validation_loader is None
    assert test_loader is not None
