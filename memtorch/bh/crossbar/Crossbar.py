import memtorch
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from enum import Enum, auto, unique
import multiprocessing as mp
import itertools


@unique
class Scheme(Enum):
    """Scheme enumeration."""
    SingleColumn = auto()
    DoubleColumn = auto()


class Crossbar():
    """Class used to model memristor crossbars.

    Parameters
    ----------
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    memristor_model_params: **kwargs
        **kwargs to instantiate the memristor model with.
    shape : tuple
        Shape of the crossbar.
    """

    def __init__(self, memristor_model, memristor_model_params, shape):
        self.time_series_resolution = memristor_model_params.get('time_series_resolution')
        self.device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
        if len(shape) == 4: # memtorch.mn.Conv2d
            self.rows = shape[0]
            self.columns = shape[1] * shape[2] * shape[3]
        elif len(shape) == 3: # memtorch.mn.Conv1d
            self.rows = shape[0]
            self.columns = shape[1] * shape[2]
        elif len(shape) == 2: # memtorch.mn.Linear
            self.columns, self.rows = shape
        else:
            raise Exception('Unsupported crossbar shape.')

        self.rows = int(self.rows)
        self.columns = int(self.columns)
        self.devices = np.empty((self.rows, self.columns), dtype=object)
        self.devices.flat = [memristor_model(**memristor_model_params) for _ in self.devices.flat]
        self.conductance_matrix = torch.zeros((self.rows, self.columns)).to(self.device)
        self.g_np = np.vectorize(lambda x: x.g)
        self.update(from_devices=True)

    def update(self, from_devices=True, parallelize=True):
        """Method to update either the layers conductance_matrix or each devices conductance state.

        Parameters
        ----------
        from_devices : bool
            The conductance matrix can either be updated from all devices (True), or each device can be updated from the conductance matrix (False).
        parallelize : bool
            The operation is parallelized (True).
        """
        if from_devices:
            self.conductance_matrix = torch.tensor(self.g_np(self.devices)).to(self.device)
        else:
            if parallelize:
                def write_conductance(device, conductance):
                    device.g = conductance

                np.frompyfunc(write_conductance, 2, 0)(self.devices, self.conductance_matrix.detach().cpu())
            else:
                for i in range(0, self.rows):
                    for j in range(0, self.columns):
                        self.devices[i][j].g = self.conductance_matrix[i][j].item()

    def write_conductance_matrix(self, conductance_matrix, transistor=True, programming_routine=None):
        """Method to directly program (alter) the conductance of all devices within the crossbar.

        Parameters
        ----------
        conductance_matrix : torch.FloatTensor
            Conductance matrix to write.
        transistor : bool
            Used to determine if a 1T1R (True) or 1R arrangement (False) is simulated.
        programming_routine
            Programming routine (method) to use.
        """
        if transistor:
            if len(conductance_matrix.shape) == 4 or len(conductance_matrix.shape) == 3: # memtorch.mn.Conv1d and memtorch.mn.Conv2d
                self.conductance_matrix = conductance_matrix.reshape(self.rows, self.columns)
            elif len(conductance_matrix.shape) == 2: # memtorch.mn.Linear
                conductance_matrix = conductance_matrix.T.clone().detach().to(self.device)
                assert(conductance_matrix.shape[0] == self.rows and conductance_matrix.shape[1] == self.columns)
                min = torch.tensor(1 / np.vectorize(lambda x: x.r_off)(self.devices)).to(self.device).float()
                max = torch.tensor(1 / np.vectorize(lambda x: x.r_on)(self.devices)).to(self.device).float()
                self.conductance_matrix = torch.max(torch.min(conductance_matrix, max), min).to(self.device)
            else:
                raise('Unsupported crossbar shape.')

            self.update(from_devices=False)
        else:
            assert programming_routine is not None, 'programming_routine must be defined if transistor is False.'
            for i in range(0, self.rows):
                for j in range(0, self.columns):
                    self.devices[i][j] = programming_routine(self, (i, j), conductance_matrix[i][j])

            self.update(from_devices=True)


def init_crossbar(weights, memristor_model, memristor_model_params, transistor, mapping_routine, programming_routine, scheme=Scheme.DoubleColumn):
    """Method to initialise and construct memristive crossbars.

    Parameters
    ----------
    weights : torch.tensor
        Weights to map.
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    memristor_model_params: **kwargs
        **kwargs to instantiate the memristor model with.
    transistor : bool
        Used to determine if a 1T1R (True) or 1R arrangement (False) is simulated.
    mapping_routine : function
        Mapping routine to use.
    programming_routine : function
        Programming routine to use.
    scheme : memtorch.bh.Scheme
        Scheme enum.

    Returns
    -------
    tuple
        The constructed crossbars and forward() function.
    """
    assert scheme in Scheme, 'scheme must be a Scheme Enum.'
    weights_ = weights.data.detach().clone()
    crossbars = []
    reference_memristor_model_params = {**memristor_model_params, **{'reference': True}}
    reference_memristor_model = memristor_model(**reference_memristor_model_params)
    if scheme == Scheme.DoubleColumn:
        crossbars.append(memtorch.bh.crossbar.Crossbar(memristor_model, memristor_model_params, weights.shape))
        crossbars.append(memtorch.bh.crossbar.Crossbar(memristor_model, memristor_model_params, weights.shape))
        pos_conductance_matrix, neg_conductance_matrix = mapping_routine(weights_,
                                                                         reference_memristor_model.r_on,
                                                                         reference_memristor_model.r_off,
                                                                         scheme)

        crossbars[0].write_conductance_matrix(pos_conductance_matrix, transistor=transistor, programming_routine=programming_routine)
        crossbars[1].write_conductance_matrix(neg_conductance_matrix, transistor=transistor, programming_routine=programming_routine)
        def out(crossbars, operation, *args):
            return operation(crossbars[0], *args) - operation(crossbars[1], *args)

    elif scheme == Scheme.SingleColumn:
        crossbars.append(memtorch.bh.crossbar.Crossbar(memristor_model, memristor_model_params, weights.shape))
        conductance_matrix = mapping_routine(weights_,
                                             reference_memristor_model.r_on,
                                             reference_memristor_model.r_off,
                                             scheme)
        crossbars[0].write_conductance_matrix(conductance_matrix, transistor=transistor, programming_routine=programming_routine)
        g_m = ((1 / reference_memristor_model.r_on) + (1 / reference_memristor_model.r_off)) / 2
        def out(crossbars, operation, *args):
            return operation(crossbars[0], *args) - g_m

    else:
        raise('%s is not currently supported.' % scheme)

    return crossbars, out

def simulate(input, devices, parallelize=False, nl=True):
    """Method to simulate non-linear IV device characterisitcs for a 2-D crossbar architecture given scaled inputs.

    Parameters
    ----------
    input : tensor
        Scaled input tensor.
    devices : numpy.ndarray
        Devices to simulate.
    parallelize : bool
        The operation is parallelized (True).
    nl : bool
        Use lookup tables rather than simulating each device (True).

    Returns
    -------
    numpy.ndarray
        Output ndarray.
    """
    input_rows, input_columns = input.shape
    devices_rows, devices_columns = devices.shape
    mat_res = np.zeros((input_rows, devices_columns))
    if parallelize:
        def pool_nl(i, j, k):
            mat_res[i][j] += devices[k][j].det_current(input[i][k].item())

        def pool_simulate(i, j, k):
            mat_res[i][j] += devices[k][j].simulate(torch.Tensor[input[i][k]], return_current=True).item()

        pool = mp.Pool()
        if nl:
            pool.map(pool_nl, itertools.product(range(input_rows), range(devices_columns), range(input_columns)))
        else:
            pool.map(pool_simulate, itertools.product(range(input_rows), range(devices_columns), range(input_columns)))
    else:
        if nl:
            for i in range(input_rows):
                for j in range(devices_columns):
                    for k in range(input_columns):
                        mat_res[i][j] += devices[k][j].det_current(input[i][k].item())
        else:
            for i in range(input_rows):
                for j in range(devices_columns):
                    for k in range(input_columns):
                        mat_res[i][j] += devices[k][j].simulate(torch.Tensor([input[i][k]]).cpu(), return_current=True).item()

    return mat_res
