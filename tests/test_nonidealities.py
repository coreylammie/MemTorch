import pytest
import numpy as np
import copy
import math
import torch
import memtorch
from debug_networks import debug_networks
from memtorch.mn.Module import patch_model
from memtorch.map.Parameter import naive_map
from memtorch.bh.crossbar.Program import naive_program
from memtorch.bh.nonideality.NonIdeality import apply_nonidealities


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
networks = debug_networks()
patched_networks = []
for network in networks:
    patched_networks.append(patch_model(network,
                                  memristor_model=memtorch.bh.memristor.LinearIonDrift,
                                  memristor_model_params={},
                                  module_parameters_to_patch=[type(network.layer)],
                                  mapping_routine=naive_map,
                                  transistor=True,
                                  programming_routine=None,
                                  scheme=memtorch.bh.Scheme.SingleColumn))

def test_device_faults():
    for patched_network in patched_networks:
        patched_network_lrs = apply_nonidealities(copy.deepcopy(patched_network),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                                  lrs_proportion=0.5,
                                  hrs_proportion=0,
                                  electroform_proportion=0)
        patched_tensor_lrs = patched_network_lrs.layer.crossbars[0].conductance_matrix.float().to(device)
        lrs = torch.tensor(1 / np.vectorize(lambda x: x.r_on)(patched_network_lrs.layer.crossbars[0].devices)).float().to(device)
        lrs_percentage = sum(torch.isclose(patched_tensor_lrs, lrs).view(-1)).item() / patched_tensor_lrs.numel()
        patched_network_hrs = apply_nonidealities(copy.deepcopy(patched_network),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                                  lrs_proportion=0,
                                  hrs_proportion=0.25,
                                  electroform_proportion=0.25)
        patched_tensor_hrs = patched_network_hrs.layer.crossbars[0].conductance_matrix.float().to(device)
        hrs = torch.tensor(1 / np.vectorize(lambda x: x.r_off)(patched_network_hrs.layer.crossbars[0].devices)).float().to(device)
        hrs_percentage = sum(torch.isclose(patched_tensor_hrs, hrs).view(-1)).item() / patched_tensor_hrs.numel()
        assert lrs_percentage >= 0.25 and hrs_percentage >= 0.25 # To account for some stochasticity

def test_finite_conductance_states(conductance_states=5):
    for patched_network in patched_networks:
        patched_network_finite_states = apply_nonidealities(copy.deepcopy(patched_network),
                                  non_idealities=[memtorch.bh.nonideality.NonIdeality.FiniteConductanceStates],
                                  conductance_states=5)
        conductance_matrix = patched_network.layer.crossbars[0].conductance_matrix
        quantized_conductance_matrix = patched_network_finite_states.layer.crossbars[0].conductance_matrix
        quantized_conductance_matrix_unique = quantized_conductance_matrix.unique()
        valid_values = torch.linspace(patched_network.layer.crossbars[0].conductance_matrix.min(),
                                    patched_network.layer.crossbars[0].conductance_matrix.max(),
                                    conductance_states)
        assert any([bool(val) for val in [torch.isclose(quantized_conductance_matrix_unique, valid_value).any() for valid_value in valid_values]])
        assert conductance_matrix.shape == quantized_conductance_matrix.shape

def test_non_linear():
    for patched_network in patched_networks:
        assert 1 == 1 # To be implemented
