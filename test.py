import torch
import memtorch
from memtorch.bh.crossbar.Tile import gen_tiles
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
import memtorch_cuda_bindings as memtorch_bindings

device = torch.device('cuda:0')
tile_shape = (4, 4)

# print('----------')
a = torch.tensor([0.1, 0.4, 18, 2.6])
b = memtorch.bh.quantize(a, 4, overflow_rate=0.0, quant_method="log")
print(b)
print('----------')
a = torch.ones(8, 8)#* 0.1
b = torch.ones(8, 8) / 1000
R_source = 5
R_line  = 5

a_tiles, a_map = gen_tiles(a, tile_shape, input=True)
b_tiles, b_map = gen_tiles(b, tile_shape, input=False)

r = memtorch_bindings.tile_matmul(a_tiles.to(device), a_map.to(device), a.shape, b_tiles.to(device), b_map.to(device), b.shape, R_source, R_line, 50)
# t = memtorch_bindings.tile_matmul(a_tiles.to(device), a_map.to(device), a.shape, b_tiles.to(device), b_map.to(device), b.shape, 50)
# print(r)
# print(t)


# linear = torch.nn.Linear(20, 10, bias=True)
# m_linear = memtorch.mn.Linear(
#     linear_layer=linear,
#     memristor_model=memtorch.bh.memristor.VTEAM,
#     memristor_model_params={'r_on': 1e2, 'r_off': 1e4},
#     mapping_routine=naive_map,
#     transistor=False,
#     programming_routine=None,
#     tile_shape=(8, 8),
#     max_input_voltage=0.3,
#     scaling_routine=naive_scale,
#     source_resistance=5,
#     line_resistance=5,
# )

# m_linear.tune()

# reference_memristor = memtorch.bh.memristor.VTEAM
# linear = memtorch.mn.Linear()