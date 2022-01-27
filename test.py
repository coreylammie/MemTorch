import torch
import memtorch
from memtorch.bh.crossbar.Tile import gen_tiles
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map
import memtorch_cuda_bindings as memtorch_bindings

device = torch.device('cuda:0')
linear = torch.nn.Linear(1000, 1000, bias=True).to(device)
m_linear = memtorch.mn.Linear(
    linear_layer=linear,
    memristor_model=memtorch.bh.memristor.VTEAM,
    memristor_model_params={'r_on': 1e5, 'r_off': 1e6},
    mapping_routine=naive_map,
    transistor=False,
    programming_routine=None,
    tile_shape=(256, 256),
    max_input_voltage=0.3,
    scaling_routine=naive_scale,
    source_resistance=2,
    line_resistance=2,
    ADC_resolution=8,
    ADC_overflow_rate=0.0,
    quant_method='linear',
)

m_linear.tune(input_shape=20)