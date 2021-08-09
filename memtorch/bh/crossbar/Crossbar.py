import itertools
import math
import multiprocessing
from enum import Enum, auto, unique

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn

import memtorch

from .Tile import gen_tiles


@unique
class Scheme(Enum):
    """Scheme enumeration."""

    SingleColumn = auto()
    DoubleColumn = auto()


class Crossbar:
    """Class used to model memristor crossbars.

    Parameters
    ----------
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    memristor_model_params: **kwargs
        **kwargs to instantiate the memristor model with.
    shape : int, int
        Shape of the crossbar.
    tile_shape : int, int
        Tile shape to use to store weights. If None, modular tiles are not used.
    use_bindings : bool
        Used to determine if C++/CUDA bindings are used (True) or not (False).
    """

    def __init__(
        self,
        memristor_model,
        memristor_model_params,
        shape,
        tile_shape=None,
        use_bindings=True,
    ):
        self.time_series_resolution = memristor_model_params.get(
            "time_series_resolution"
        )
        self.device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
        self.tile_shape = tile_shape
        self.use_bindings = use_bindings
        if hasattr(memristor_model_params, "r_off"):
            self.r_off_mean = memristor_model_params["r_off"]
            if callable(self.r_off_mean):
                self.r_off_mean = self.r_off_mean()
        else:
            self.r_off_mean = memristor_model().r_off

        if hasattr(memristor_model_params, "r_on"):
            self.r_on_mean = memristor_model_params["r_on"]
            if callable(self.r_on_mean):
                self.r_on_mean = self.r_on_mean()
        else:
            self.r_on_mean = memristor_model().r_on

        if len(shape) == 4:  # memtorch.mn.Conv2d and memtorch.mn.Conv3d
            self.rows = shape[1] * shape[2] * shape[3]
            self.columns = shape[0]
        elif len(shape) == 3:  # memtorch.mn.Conv1d
            self.rows = shape[1] * shape[2]
            self.columns = shape[0]
        elif len(shape) == 2:  # memtorch.mn.Linear
            self.columns, self.rows = shape
        else:
            raise Exception("Unsupported crossbar shape.")

        self.rows = int(self.rows)
        self.columns = int(self.columns)
        if tile_shape is not None:
            self.tiles_map = None
            tiles_num = math.ceil(self.rows / tile_shape[0]) * math.ceil(
                self.columns / tile_shape[1]
            )
            self.devices = np.empty(
                (tiles_num, tile_shape[0], tile_shape[1]), dtype=object
            )
            self.devices.flat = [
                memristor_model(**memristor_model_params) for _ in self.devices.flat
            ]
            self.conductance_matrix = torch.zeros(
                (tiles_num, tile_shape[0], tile_shape[1])
            )
        else:
            self.devices = np.empty((self.rows, self.columns), dtype=object)
            self.devices.flat = [
                memristor_model(**memristor_model_params) for _ in self.devices.flat
            ]
            self.conductance_matrix = torch.zeros((self.rows, self.columns)).to(
                self.device
            )

        self.g_np = np.vectorize(lambda x: x.g)
        self.update(from_devices=False)
        self.max_abs_conductance = torch.abs(self.conductance_matrix).flatten()

    def update(self, from_devices=True, parallelize=False):
        """Method to update either the layers conductance_matrix or each devices conductance state.

        Parameters
        ----------
        from_devices : bool
            The conductance matrix can either be updated from all devices (True), or each device can be updated from the conductance matrix (False).
        parallelize : bool
            The operation is parallelized (True).
        """
        if from_devices:
            self.conductance_matrix = torch.tensor(self.g_np(self.devices)).to(
                self.device
            )
            self.max_abs_conductance = (
                torch.abs(self.conductance_matrix).flatten().max()
            )
        else:
            if parallelize:

                def write_conductance(device, conductance):
                    device.set_conductance(conductance)

                np.frompyfunc(write_conductance, 2, 0)(
                    self.devices, self.conductance_matrix.detach().cpu()
                )
            else:
                if self.tile_shape is not None:
                    for i in range(0, self.devices.shape[0]):
                        for j in range(0, self.devices.shape[1]):
                            for k in range(0, self.devices.shape[2]):
                                self.devices[i][j][k].set_conductance(
                                    self.conductance_matrix[i][j][k].item()
                                )
                else:
                    for i in range(0, self.rows):
                        for j in range(0, self.columns):
                            self.devices[i][j].set_conductance(
                                self.conductance_matrix[i][j].item()
                            )

    def write_conductance_matrix(
        self,
        conductance_matrix,
        transistor=True,
        programming_routine=None,
        programming_routine_params={},
    ):
        """Method to directly program (alter) the conductance of all devices within the crossbar.

        Parameters
        ----------
        conductance_matrix : torch.FloatTensor
            Conductance matrix to write.
        transistor : bool
            Used to determine if a 1T1R (True) or 0T1R arrangement (False) is simulated.
        programming_routine
            Programming routine (method) to use.
        programming_routine_params : **kwargs
            Programming routine keyword arguments.
        """
        if (
            len(conductance_matrix.shape) == 3 or len(conductance_matrix.shape) == 4
        ):  # memtorch.mn.Conv1d, memtorch.mn.Conv2d, and memtorch.mn.Conv3d
            conductance_matrix = conductance_matrix.reshape(self.columns, self.rows).T
        elif len(conductance_matrix.shape) == 2:  # memtorch.mn.Linear
            conductance_matrix = conductance_matrix.T.clone().detach().to(self.device)
            assert (
                conductance_matrix.shape[0] == self.rows
                and conductance_matrix.shape[1] == self.columns
            )
        else:
            raise Exception("Unsupported crossbar shape.")

        if self.tile_shape is not None:
            conductance_matrix, tiles_map = gen_tiles(
                conductance_matrix,
                self.tile_shape,
                input=False,
                use_bindings=self.use_bindings,
            )
            self.tiles_map = tiles_map

        min = (
            torch.tensor(1 / np.vectorize(lambda x: x.r_off)(self.devices))
            .to(self.device)
            .float()
        )
        max = (
            torch.tensor(1 / np.vectorize(lambda x: x.r_on)(self.devices))
            .to(self.device)
            .float()
        )
        conductance_matrix = torch.max(
            torch.min(conductance_matrix.to(self.device), max), min
        )
        if transistor:
            self.conductance_matrix = conductance_matrix
            self.max_abs_conductance = (
                torch.abs(self.conductance_matrix).flatten().max()
            )
            self.update(from_devices=False)
        else:
            assert (
                programming_routine is not None
            ), "programming_routine must be defined if transistor is False."
            if self.tile_shape is not None:
                for i in range(0, self.devices.shape[0]):
                    for j in range(0, self.devices.shape[1]):
                        for k in range(0, self.devices.shape[2]):
                            self.devices = programming_routine(
                                self,
                                (i, j, k),
                                conductance_matrix[i][j][k],
                                **programming_routine_params
                            )
            else:
                for i in range(0, self.rows):
                    for j in range(0, self.columns):
                        self.devices = programming_routine(
                            self,
                            (i, j),
                            conductance_matrix[i][j],
                            **programming_routine_params
                        )

            self.update(from_devices=True)


def init_crossbar(
    weights,
    memristor_model,
    memristor_model_params,
    transistor,
    mapping_routine,
    programming_routine,
    programming_routine_params={},
    p_l=None,
    scheme=Scheme.DoubleColumn,
    tile_shape=(128, 128),
    use_bindings=True,
):
    """Method to initialise and construct memristive crossbars.

    Parameters
    ----------
    weights : torch.Tensor
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
    programming_routine_params : **kwargs
        Programming routine keyword arguments.
    p_l: float
        If not None, the proportion of weights to retain.
    scheme : memtorch.bh.Scheme
        Scheme enum.
    tile_shape : int, int
        Tile shape to use to store weights. If None, modular tiles are not used.
    use_bindings : bool
        Used to determine if C++/CUDA bindings are used (True) or not (False).

    Returns
    -------
    tuple
        The constructed crossbars and forward() function.
    """
    assert scheme in Scheme, "scheme must be a Scheme Enum."
    weights_ = weights.data.detach().clone()
    crossbars = []
    reference_memristor_model_params = {**memristor_model_params, **{"reference": True}}
    reference_memristor_model = memristor_model(**reference_memristor_model_params)
    if scheme == Scheme.DoubleColumn:
        if len(weights.shape) == 5:  # memtorch.mn.Conv3d
            channel_idx = 0
            for channel in range(weights.shape[1]):
                channel_weights = weights.detach().clone()[:, channel, :, :, :]
                crossbars.append(
                    memtorch.bh.crossbar.Crossbar(
                        memristor_model,
                        memristor_model_params,
                        channel_weights.shape,
                        tile_shape,
                        use_bindings=use_bindings,
                    )
                )
                crossbars.append(
                    memtorch.bh.crossbar.Crossbar(
                        memristor_model,
                        memristor_model_params,
                        channel_weights.shape,
                        tile_shape,
                        use_bindings=use_bindings,
                    )
                )
                pos_conductance_matrix, neg_conductance_matrix = mapping_routine(
                    channel_weights,
                    reference_memristor_model.r_on,
                    reference_memristor_model.r_off,
                    scheme=scheme,
                    p_l=p_l,
                )
                crossbars[channel_idx].write_conductance_matrix(
                    pos_conductance_matrix,
                    transistor=transistor,
                    programming_routine=programming_routine,
                    programming_routine_params=programming_routine_params,
                )
                crossbars[channel_idx + 1].write_conductance_matrix(
                    neg_conductance_matrix,
                    transistor=transistor,
                    programming_routine=programming_routine,
                    programming_routine_params=programming_routine_params,
                )
                channel_idx += 2
        else:
            crossbars.append(
                memtorch.bh.crossbar.Crossbar(
                    memristor_model,
                    memristor_model_params,
                    weights.shape,
                    tile_shape,
                    use_bindings=use_bindings,
                )
            )
            crossbars.append(
                memtorch.bh.crossbar.Crossbar(
                    memristor_model,
                    memristor_model_params,
                    weights.shape,
                    tile_shape,
                    use_bindings=use_bindings,
                )
            )
            pos_conductance_matrix, neg_conductance_matrix = mapping_routine(
                weights_,
                reference_memristor_model.r_on,
                reference_memristor_model.r_off,
                scheme=scheme,
                p_l=p_l,
            )
            crossbars[0].write_conductance_matrix(
                pos_conductance_matrix,
                transistor=transistor,
                programming_routine=programming_routine,
                programming_routine_params=programming_routine_params,
            )
            crossbars[1].write_conductance_matrix(
                neg_conductance_matrix,
                transistor=transistor,
                programming_routine=programming_routine,
                programming_routine_params=programming_routine_params,
            )

        def out(crossbars, operation, idx=(0, 1), **kwargs):
            assert (
                len(idx) == 2
            ), "idx must contain indicies of the positive and negative crossbars"
            return operation(crossbars[idx[0]], **kwargs) - operation(
                crossbars[idx[1]], **kwargs
            )

    elif scheme == Scheme.SingleColumn:
        if len(weights.shape) == 5:  # memtorch.mn.Conv3d
            channel_idx = 0
            for channel in range(weights.shape[1]):
                channel_weights = weights.detach().clone()[:, channel, :, :, :]
                crossbars.append(
                    memtorch.bh.crossbar.Crossbar(
                        memristor_model,
                        memristor_model_params,
                        channel_weights.shape,
                        tile_shape,
                        use_bindings=use_bindings,
                    )
                )
                conductance_matrix = mapping_routine(
                    channel_weights,
                    reference_memristor_model.r_on,
                    reference_memristor_model.r_off,
                    scheme=scheme,
                    p_l=p_l,
                )
                crossbars[channel_idx].write_conductance_matrix(
                    conductance_matrix,
                    transistor=transistor,
                    programming_routine=programming_routine,
                    programming_routine_params=programming_routine_params,
                )
                channel_idx += 1
        else:
            crossbars.append(
                memtorch.bh.crossbar.Crossbar(
                    memristor_model,
                    memristor_model_params,
                    weights.shape,
                    tile_shape,
                    use_bindings=use_bindings,
                )
            )
            conductance_matrix = mapping_routine(
                weights_,
                reference_memristor_model.r_on,
                reference_memristor_model.r_off,
                scheme=scheme,
                p_l=p_l,
            )
            crossbars[0].write_conductance_matrix(
                conductance_matrix,
                transistor=transistor,
                programming_routine=programming_routine,
                programming_routine_params=programming_routine_params,
            )

        g_m = (
            (1 / reference_memristor_model.r_on) + (1 / reference_memristor_model.r_off)
        ) / 2

        def out(crossbars, operation, idx=0, **kwargs):
            return operation(crossbars[idx], **kwargs) - g_m

    else:
        raise ("%s is not currently supported." % scheme)

    return crossbars, out


def simulate_matmul(
    input,
    crossbar,
    nl=True,
    tiles_map=None,
    crossbar_shape=None,
    max_input_voltage=None,
    ADC_resolution=None,
    ADC_overflow_rate=0.0,
    quant_method=None,
    use_bindings=True,
):
    """Method to simulate non-linear IV device characterisitcs for a 2-D crossbar architecture given scaled inputs.

    Parameters
    ----------
    input : torch.Tensor
        Scaled input tensor.
    crossbar : memtorch.bh.Crossbar
        Crossbar containing devices to simulate.
    nl : bool
        Use lookup tables rather than simulating each device (True).
    tiles_map: torch.Tensor
        Tiles map for devices if tile_shape is not None.
    crossbar_shape : int, int
        Crossbar shape if tile_shape is not None.
    max_input_voltage : float
        Maximum input voltage used to encode inputs. If None, inputs are unbounded.
    ADC_resolution : int
        ADC resolution (bit width). If None, quantization noise is not accounted for.
    ADC_overflow_rate : float
        Overflow rate threshold for linear quanitzation (if ADC_resolution is not None).
    quant_method:
        Quantization method. Must be in memtorch.bh.Quantize.quant_methods.
    use_bindings : bool
        Used to determine if C++/CUDA bindings are used (True) or not (False).

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    devices = crossbar.devices
    if max_input_voltage is not None:
        output_max = crossbar.max_abs_conductance * max_input_voltage
    else:
        output_max = float("inf")

    del crossbar
    assert len(devices.shape) == 2 or len(devices.shape) == 3, "Invalid devices shape."
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
    if quant_method is not None:
        assert (
            ADC_resolution is not None
            and type(ADC_resolution) == int
            and ADC_resolution > 0
        ), "ADC resolution is invalid."
        assert (
            quant_method in memtorch.bh.Quantize.quant_methods
        ), "quant_method is not valid."
        assert (
            ADC_overflow_rate is not None
        ), "ADC_overflow_rate must be specified if quant_method is not None."

    input_rows, input_columns = input.shape
    if len(devices.shape) == 2:
        devices_rows, devices_columns = devices.shape
        mat_res_ = torch.zeros((input_rows, devices_columns)).to(device)
        if nl:
            for i in range(input_rows):
                for j in range(devices_columns):
                    for k in range(input_columns):
                        mat_res_[i][j] += devices[k][j].g * input[i][k]
        else:
            for i in range(input_rows):
                for j in range(devices_columns):
                    for k in range(input_columns):
                        mat_res_[i][j] += (
                            devices[k][j]
                            .simulate(
                                torch.Tensor([input[i][k]]).cpu(), return_current=True
                            )
                            .item()
                        )

        mat_res_ = torch.clamp(mat_res_, min=-output_max, max=output_max)
        if quant_method is not None:
            mat_res_ = memtorch.bh.Quantize.quantize(
                mat_res_,
                quant=ADC_resolution,
                overflow_rate=ADC_overflow_rate,
                quant_method=quant_method,
            )
    else:
        assert (
            tiles_map is not None and crossbar_shape is not None
        ), "tiles_map is not None."
        tile_shape = devices.shape[-2:]
        input_tiles, input_tiles_map = gen_tiles(
            input, tile_shape, input=True, use_bindings=use_bindings
        )
        mat_res_ = torch.zeros((input.shape[0], crossbar_shape[1])).to(device)

        def tile_simulate_matmul_row(
            input_row_tiles,
            input_tiles_map,
            devices,
            tiles_map,
            crossbar_shape,
            nl,
            ADC_resolution,
            ADC_overflow_rate,
            quant_method,
        ):
            device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")
            tile_shape = devices.shape[-2:]
            partial_sum = torch.zeros((tiles_map.shape[1], tile_shape[1])).to(device)
            for j in range(tiles_map.shape[1]):
                for i in range(tiles_map.shape[0]):
                    tile_a = input_row_tiles[int(input_tiles_map[i])]
                    if len(tile_a.shape) == 1:
                        tile_a = tile_a.unsqueeze(0)

                    tile_b = devices[int(tiles_map[i][j])]
                    mat_res = torch.zeros((tile_a.shape[0], tile_b.shape[1])).to(device)
                    for ii in range(tile_a.shape[0]):
                        for jj in range(tile_b.shape[1]):
                            for kk in range(tile_b.shape[0]):
                                if nl:
                                    mat_res[ii][jj] += (
                                        tile_a[ii][kk].item() * tile_b[kk][jj].g
                                    )
                                else:
                                    mat_res[ii][jj] += (
                                        tile_b[kk][jj]
                                        .simulate(
                                            torch.Tensor([tile_a[ii][kk]]).cpu(),
                                            return_current=True,
                                        )
                                        .item()
                                    )

                    mat_res = torch.clamp(mat_res, min=-output_max, max=output_max)
                    if quant_method is not None:
                        partial_sum[j] += memtorch.bh.Quantize.quantize(
                            mat_res.squeeze(),
                            quant=ADC_resolution,
                            overflow_rate=ADC_overflow_rate,
                            quant_method=quant_method,
                        )
                    else:
                        partial_sum[j] += mat_res.squeeze()

            output_act = partial_sum.flatten()
            output_act = output_act[: crossbar_shape[1]]
            return output_act

        if input_tiles.shape[-2] > 1:
            for row_idx in range(input_tiles.shape[-2]):
                mat_res_[row_idx] = tile_simulate_matmul_row(
                    input_tiles[:, row_idx, :],
                    input_tiles_map,
                    devices,
                    tiles_map,
                    crossbar_shape,
                    nl,
                    ADC_resolution,
                    ADC_overflow_rate,
                    quant_method,
                )
        else:
            mat_res_ = tile_simulate_matmul_row(
                input_tiles,
                input_tiles_map,
                devices,
                tiles_map,
                crossbar_shape,
                nl,
                ADC_resolution,
                ADC_overflow_rate,
                quant_method,
            )

    return mat_res_
