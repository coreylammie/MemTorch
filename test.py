import torch
import memtorch
from memtorch.bh.crossbar.Tile import gen_tiles
import memtorch_cuda_bindings as memtorch_bindings

device = torch.device('cuda:0')
tile_shape = (5, 4)
a = torch.ones((1, tile_shape[0])) * 2
b = torch.ones(tile_shape) * 100
R_source = 5
R_line  = 5

a_tiles, a_map = gen_tiles(a, tile_shape, input=True)
b_tiles, b_map = gen_tiles(b, tile_shape, input=False)
print('----------')
r = memtorch_bindings.tile_matmul(a_tiles.to(device), a_map.to(device), a.shape, b_tiles.to(device), b_map.to(device), b.shape, R_source, R_line, 50)
print(r)
# print(a)
# print(b)
# print(torch.matmul(a, b))
# Expected result is [0.202456373855409	0.126880152953024 0.0947633761059608 0.0822822374841270]