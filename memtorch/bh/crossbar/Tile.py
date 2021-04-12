# Modular tile implementation based on: https://github.com/xxwang1/DNN-accelerator-based-on-tiled-architecture
import math

import numpy as np
import torch
import torch.nn as nn

import memtorch


class Tile:
    """Class used to create modular crossbar tiles to represent 2D matrices.

    Parameters
    ----------
    tile_shape : (int, int)
        Tile shape to use to store weights.
    patch_num : int
        Patch number.
    """

    def __init__(self, tile_shape, patch_num=None):
        self.tile_shape = tile_shape
        self.patch_num = patch_num
        if patch_num is None:
            self.array = torch.zeros(tile_shape)
        else:
            self.array = torch.zeros((patch_num, tile_shape[0]))

    def update_array(self, new_array):
        """Method to update the tile's weights.

        Parameters
        ----------
        new_array : torch.tensor
            New array to construct the tile with.
        """
        if new_array.shape == self.tile_shape or new_array.shape == (
            self.patch_num,
            self.tile_shape[0],
        ):
            self.array = new_array
        else:
            new_col_cnt = new_array.shape[1]
            if type(new_array) == np.ndarray:
                new_array = torch.from_numpy(new_array)
            else:
                new_array = new_array.clone().detach()

            if self.patch_num is None:
                new_row_cnt = new_array.shape[0]
                self.array[:new_row_cnt, :new_col_cnt] = new_array
            else:
                self.array[:, :new_col_cnt] = new_array


def gen_tiles(tensor, tile_shape, input=False):
    """Method to generate a set of modular tiles representative of a tensor.

    Parameters
    ----------
    tensor : torch.tensor
        TBD.
    tile_shape : (int, int)
        Tile shape to use to store weights.
    input : bool
        Used to determine if a tensor is an input (True).

    Returns
    -------
    (torch.tensor, torch.tensor)
        Tiles and tile_map.
    """
    tiles = []
    tensor_shape = tensor.shape
    if input:
        patch_num = tensor_shape[0]
        tile_columns = math.ceil(
            tensor_shape[1] / tile_shape[0]
        )  # Number of mapped arrays
        tiles_map = torch.empty([tile_columns])
        for tile_column in range(tile_columns):
            tiles.append(Tile(patch_num=patch_num, tile_shape=tile_shape))
            column_start = (
                tile_column * tile_shape[0]
            )  # Set the range of the array slice by defining starting and ending columns
            if tile_column == tile_columns - 1:  # Execute if last column
                column_end = -1
            else:
                column_end = (tile_column + 1) * tile_shape[0]

            if column_end == -1:  # If the last column
                tiles[-1].update_array(tensor[:, column_start:])
            else:
                tiles[-1].update_array(tensor[:, column_start:(column_end)])

            new_tile_id = len(tiles) - 1
            tiles_map[tile_column] = new_tile_id
    else:
        tile_rows = math.ceil(tensor_shape[0] / tile_shape[0])
        tile_columns = math.ceil(tensor_shape[1] / tile_shape[1])
        tiles_map = torch.empty([tile_rows, tile_columns])
        for tile_row in range(tile_rows):
            row_start = tile_row * tile_shape[0]
            if tile_row == tile_rows - 1:  # Execute if last row
                row_end = -1
            else:
                row_end = (tile_row + 1) * tile_shape[0]

            for tile_column in range(tile_columns):
                tiles.append(Tile(tile_shape=tile_shape))
                column_start = (
                    tile_column * tile_shape[1]
                )  # Set the range of the array slice by defining starting and ending columns
                if tile_column == tile_columns - 1:  # Execute if last column
                    column_end = -1
                else:
                    column_end = (tile_column + 1) * tile_shape[1]

                if row_end == -1 and column_end == -1:  # If last row and last column
                    tiles[-1].update_array(tensor[row_start:, column_start:])
                elif (
                    row_end == -1 and column_end != -1
                ):  # If last row but not last column
                    tiles[-1].update_array(tensor[row_start:, column_start:column_end])
                elif (
                    row_end != -1 and column_end == -1
                ):  # If last column but not last row
                    tiles[-1].update_array(tensor[row_start:row_end, column_start:])
                else:  # If neither last row nor last column
                    tiles[-1].update_array(
                        tensor[row_start:(row_end), column_start:(column_end)]
                    )

                new_tile_id = len(tiles) - 1
                tiles_map[tile_row][tile_column] = new_tile_id

    tiles = torch.tensor([np.array(tile.array.detach().cpu()) for tile in tiles])
    return tiles, tiles_map


def tile_matmul(
    mat_a_tiles,
    mat_a_tiles_map,
    mat_a_shape,
    mat_b_tiles,
    mat_b_tiles_map,
    mat_b_shape,
    ADC_resolution=None,
    ADC_overflow_rate=0.0,
    quant_method=None,
):
    """Method to perform 2D matrix multiplication, given two sets of tiles.

    Parameters
    ----------
    mat_a_tiles : torch.tensor
        Tiles representing matrix A.
    mat_a_tiles_map : torch.tensor
        Tiles map for matrix A.
    mat_a_shape : (int, int)
        Shape of matrix A.
    mat_b_tiles : torch.tensor
        Tiles representing matrix B.
    mat_b_tiles_map : torch.tensor
        Tiles map for matrix B.
    mat_b_shape : (int, int)
        Shape of matrix B.
    ADC_resolution : int
        ADC resolution (bit width). If None, quantization noise is not accounted for.
    ADC_overflow_rate : float
        Overflow rate threshold for linear quanitzation (if ADC_resolution is not None).
    quant_method:
        Quantization method. Must be in ['linear', 'log', 'log_minmax', 'minmax', 'tanh'], or None.

    Returns
    -------
    torch.tensor
        Output tensor.
    """

    def tile_matmul_row(
        mat_a_row_tiles,
        mat_a_tiles_map,
        mat_a_shape,
        mat_b_tiles,
        mat_b_tiles_map,
        mat_b_shape,
        ADC_resolution=None,
        ADC_overflow_rate=0.0,
        quant_method=None,
    ):
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

        tile_shape = mat_b_tiles.shape[-2:]
        partial_sum = torch.zeros((mat_b_tiles_map.shape[1], tile_shape[1])).to(device)
        for j in range(mat_b_tiles_map.shape[1]):
            for i in range(mat_b_tiles_map.shape[0]):
                tile_a = mat_a_row_tiles[int(mat_a_tiles_map[i])]
                tile_b = mat_b_tiles[int(mat_b_tiles_map[i][j])]
                if quant_method is not None:
                    partial_sum[j] += memtorch.bh.Quantize.quantize(
                        torch.matmul(tile_a.to(device), tile_b.to(device)).squeeze(),
                        bits=ADC_resolution,
                        overflow_rate=ADC_overflow_rate,
                        quant_method=quant_method,
                    )
                else:
                    partial_sum[j] += torch.matmul(
                        tile_a.to(device), tile_b.to(device)
                    ).squeeze()

        output_act = partial_sum.flatten()
        output_act = output_act[: mat_b_shape[1]]
        return output_act

    assert (
        mat_a_tiles.shape[-1] == mat_b_tiles.shape[-2]
        and len(mat_a_tiles.shape) == 3
        and len(mat_b_tiles.shape) == 3
        and mat_a_tiles.shape[-2] != 0
    ), "Incompatible tile shapes used."
    result = torch.zeros((mat_a_shape[0], mat_b_shape[1]))
    if mat_a_tiles.shape[-2] > 1:
        for row_idx in range(mat_a_tiles.shape[-2]):
            result[row_idx] = tile_matmul_row(
                mat_a_tiles[:, row_idx, :],
                mat_a_tiles_map,
                mat_a_shape,
                mat_b_tiles,
                mat_b_tiles_map,
                mat_b_shape,
                ADC_resolution,
                ADC_overflow_rate,
                quant_method,
            )
    else:
        result = tile_matmul_row(
            mat_a_tiles,
            mat_a_tiles_map,
            mat_a_shape,
            mat_b_tiles,
            mat_b_tiles_map,
            mat_b_shape,
            ADC_resolution,
            ADC_overflow_rate,
            quant_method,
        )

    return result
