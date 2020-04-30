import torch
import memtorch
import numpy as np


def apply_device_faults(layer, lrs_proportion, hrs_proportion, electroform_proportion):
    """Method to model device failure within a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    lrs_proportion : float
        Proportion of devices which become stuck at a low resistance state.
    hrs_proportion : float
        Proportion of devices which become stuck at a high resistance state.
    electroform_proportion : float
        Proportion of devices which fail to electroform.

    Returns
    -------
    memtorch.mn
        The patched memristive layer.
    """
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    def apply_device_faults_to_crossbar(crossbar, lrs_proportion, hrs_proportion):
        crossbar_min = torch.tensor(1 / (np.vectorize(lambda x: x.r_off)(crossbar.devices))).view(-1).to(device)
        crossbar_max = torch.tensor(1 / (np.vectorize(lambda x: x.r_on)(crossbar.devices))).view(-1).to(device)
        crossbar_min_indices = np.random.choice(np.arange(torch.numel(crossbar_min)), replace=False, size=int(torch.numel(crossbar_min) * hrs_proportion))
        crossbar_shape = crossbar.conductance_matrix.shape
        crossbar.conductance_matrix = crossbar.conductance_matrix.view(-1)
        for index in crossbar_min_indices:
            crossbar.conductance_matrix[index] = crossbar_min[index]

        crossbar_max_indices = np.random.choice(np.arange(torch.numel(crossbar_max)), replace=False, size=int(torch.numel(crossbar_max) * lrs_proportion))
        for index in crossbar_max_indices:
            crossbar.conductance_matrix[index] = crossbar_max[index]

        crossbar.conductance_matrix = crossbar.conductance_matrix.view(crossbar_shape)
        return crossbar

    hrs_proportion = hrs_proportion + electroform_proportion
    for i in range(len(layer.crossbars)):
        layer.crossbars[i] = apply_device_faults_to_crossbar(layer.crossbars[i], lrs_proportion, hrs_proportion)

    return layer
