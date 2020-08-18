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

def apply_cycle_variability(layer, distribution=torch.distributions.normal.Normal, min=0, max=float('Inf'), parallelize=False, **kwargs):
    """Method to apply cycle-to-cycle variability to a memristive layer.

    Parameters
    ----------
    layer : memtorch.mn
        A memrstive layer.
    distribution : torch.distributions
        torch distribution.
    min : float
        Minimum value to sample.
    max: float
        Maximum value to sample.
    parallelize : bool
        The operation is parallelized (True).
    """
    device = torch.device('cpu' if 'cpu' in memtorch.__version__ else 'cuda')
    def apply_cycle_variability_to_crossbar(crossbar, distribution=torch.distributions.normal.Normal, min=0, max=float('Inf'), parallelize=False, **kwargs):
        assert distribution == torch.distributions.normal.Normal, 'Currently, only torch.distributions.normal.Normal is supported.'
        assert kwargs['std'] is not None, 'std must be defined when distribution=torch.distributions.normal.Normal.'
        print(type(crossbar))
        shape = crossbar.conductance_matrix.shape
        r_off_m = distribution(torch.zeros(shape).fill_(crossbar.r_off_mean), torch.zeros(shape).fill_(kwargs['std'] * 2))
        r_on_m = distribution(torch.zeros(shape).fill_(crossbar.r_on_mean), torch.zeros(shape).fill_(kwargs['std']))
        r_off = r_off_m.sample().clamp(min, max)
        r_on = r_on_m.sample().clamp(min, max)
        if parallelize:
            def write_r_off(device, conductance):
                device.r_off(conductance)

            def write_r_on(device, conductance):
                device.r_on(conductance)

            np.frompyfunc(write_r_off, 2, 0)(crossbar.devices, r_off.item())
            np.frompyfunc(write_r_on, 2, 0)(crossbar.devices, r_on.item())
        else:
            for i in range(0, crossbar.rows):
                for j in range(0, crossbar.columns):
                    crossbar.devices[i][j].r_off = r_off[i][j].item()
                    crossbar.devices[i][j].r_on = r_on[i][j].item()

        crossbar.conductance_matrix = torch.max(torch.min(crossbar.conductance_matrix.clone().detach().to(device), 1 / r_on), 1 / r_off)
        crossbar.update(from_devices=False)
        return crossbar

    for i in range(len(layer.crossbars)):
        layer.crossbars[i] = apply_cycle_variability_to_crossbar(layer.crossbars[i], distribution, min=min, max=max, **kwargs)

    return layer
