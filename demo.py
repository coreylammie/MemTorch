import torch
import memtorch

memristor_model = memtorch.bh.memristor.LinearIonDrift
memristor_model_params = {'r_off': memtorch.bh.StochasticParameter(loc=200, scale=20, min=2), 'r_on': memtorch.bh.StochasticParameter(loc=100, scale=10, min=1)}
conv_layer = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
memristive_conv_layer = memtorch.mn.Conv2d(conv_layer, memristor_model, memristor_model_params, transistor=True, scheme=memtorch.bh.Scheme.SinglColumn)
memristive_conv_layer.tune()
memristor_model().plot_hysteresis_loop()
