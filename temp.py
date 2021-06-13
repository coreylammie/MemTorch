import torch
import memtorch
import memtorch_bindings
from memtorch.bh.crossbar.Tile import gen_tiles

tile_shape = (16, 16)
test_shape = (25, 25)
a = torch.zeros(test_shape).uniform_(0, 1)
b = torch.zeros(test_shape).uniform_(0, 1)

tile_a_tiles, tile_a_map = gen_tiles(a, tile_shape, input=True)
tile_b_tiles, tile_b_map = gen_tiles(b, tile_shape, input=False)

print(tile_b_tiles.shape[-2:])
memtorch_bindings.tile_matmul(
    tile_a_tiles, tile_a_map, test_shape, tile_b_tiles, tile_a_map, test_shape
)
