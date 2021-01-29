import memtorch
import torch
import torch.nn as nn
import numpy as np
import math


class Tile:
    def __init__(self, tile_shape, patch_num=None):
        self.tile_shape = tile_shape
        self.patch_num = patch_num
        if patch_num is None:
            self.array = torch.zeros(tile_shape)
        else:
            self.array = torch.zeros((patch_num, tile_shape[0]))

    def update_array(self, new_array):
        if new_array.shape == self.tile_shape or new_array.shape == (self.patch_num, self.tile_shape[0]):
            self.array = new_array
        else:
            new_col_cnt = new_array.shape[1]
            if self.patch_num is None:
                new_row_cnt = new_array.shape[0]
                self.array[:new_row_cnt, : new_col_cnt] = torch.tensor(new_array)
            else:
                self.array[:, :new_col_cnt] = torch.tensor(new_array)

def gen_tiles(tensor, tile_shape, input=False):
    """ Method to generate a set of modular tiles representative of a tensor."""
    # if len(tensor.shape) == 1:
    #     tensor = tensor.unsqueeze(1)

    tiles = []
    tensor_shape = tensor.shape
    if input:
        patch_num = tensor_shape[0]
        tile_columns = math.ceil(tensor_shape[1] / tile_shape[0]) # Number of mapped arrays

        tiles_map = torch.empty([tile_columns])
        for tile_column in range(tile_columns):
            tiles.append(Tile(patch_num=patch_num, tile_shape=tile_shape))
            column_start = tile_column * tile_shape[0] # Set the range of the array slice by defining starting and ending columns
            if tile_column == tile_columns - 1: # Execute if last column
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
            if tile_row == tile_rows - 1: # Execute if last row
                row_end = -1
            else:
                row_end = (tile_row + 1) * tile_shape[0]

            for tile_column in range(tile_columns):
                tiles.append(Tile(tile_shape=tile_shape))
                column_start = tile_column * tile_shape[1] # Set the range of the array slice by defining starting and ending columns
                if tile_column == tile_columns - 1: # Execute if last column
                    column_end = -1
                else:
                    column_end = (tile_column+1)*tile_shape[1]

                if row_end == -1 and column_end == -1: # If last row and last column
                    tiles[-1].update_array(tensor[row_start:, column_start:])
                elif row_end == -1 and column_end != -1:   # If last row but not last column
                    tiles[-1].update_array(tensor[row_start:, column_start:column_end])
                elif row_end != -1 and column_end == -1:   # If last column but not last row
                    tiles[-1].update_array(tensor[row_start:row_end, column_start:])
                else:  # If neither last row nor last column
                    tiles[-1].update_array(tensor[row_start:(row_end), column_start:(column_end)])

                new_tile_id = len(tiles)-1
                tiles_map[tile_row][tile_column] = new_tile_id

    tiles = torch.tensor([np.array(tile.array) for tile in tiles])
    return tiles, tiles_map

def tile_matmul(mat_a_tiles, mat_a_tiles_map, mat_a_shape, mat_b_tiles, mat_b_tiles_map, mat_b_shape):
    """ Method to perform 2D matrix multiplication, given two sets of tiles."""
    def tile_matmul_row(mat_a_row_tiles, mat_a_tiles_map, mat_a_shape, mat_b_tiles, mat_b_tiles_map, mat_b_shape):
        """ Method to perform 2D matrix multiplication, given two sets of tiles, where the first input is a singular row."""
        tile_shape = mat_b_tiles.shape[-2:]
        partial_sum = torch.zeros((mat_b_tiles_map.shape[1], tile_shape[1]))
        for j in range(mat_b_tiles_map.shape[1]):
            for i in range(mat_b_tiles_map.shape[0]):
                tile_a = mat_a_row_tiles[int(mat_a_tiles_map[i])]
                tile_b = mat_b_tiles[int(mat_b_tiles_map[i][j])]
                partial_sum[j] += torch.matmul(tile_a, tile_b).squeeze()

        output_act = partial_sum.flatten()
        output_act = output_act[:mat_b_shape[1]]
        return output_act

    assert mat_a_tiles.shape[-1] == mat_b_tiles.shape[-2] and len(mat_a_tiles.shape) == 3 and len(mat_b_tiles.shape) == 3 and mat_a_tiles.shape[-2] != 0, 'Incompatible tile shapes used.'
    result = torch.zeros((mat_a_shape[0], mat_b_shape[1]))
    if mat_a_tiles.shape[-2] > 1:
        for row_idx in range(mat_a_tiles.shape[-2]):
            result[row_idx] = tile_matmul_row(mat_a_tiles[:, row_idx, :], mat_a_tiles_map, mat_a_shape, mat_b_tiles, mat_b_tiles_map, mat_b_shape)
    else:
        result = tile_matmul_row(mat_a_tiles, mat_a_tiles_map, mat_a_shape, mat_b_tiles, mat_b_tiles_map, mat_b_shape)

    return result
