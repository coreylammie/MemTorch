import inspect

import numpy as np
import torch

import memtorch


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
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")

    def apply_device_faults_to_crossbar(crossbar, lrs_proportion, hrs_proportion):
        crossbar_min = (
            torch.tensor(1 / (np.vectorize(lambda x: x.r_off)(crossbar.devices)))
            .view(-1)
            .to(device)
        )
        crossbar_max = (
            torch.tensor(1 / (np.vectorize(lambda x: x.r_on)(crossbar.devices)))
            .view(-1)
            .to(device)
        )
        crossbar_min_indices = np.random.choice(
            np.arange(torch.numel(crossbar_min)),
            replace=False,
            size=int(torch.numel(crossbar_min) * hrs_proportion),
        )
        crossbar_shape = crossbar.conductance_matrix.shape
        crossbar.conductance_matrix = crossbar.conductance_matrix.reshape(-1)
        for index in crossbar_min_indices:
            crossbar.conductance_matrix[index] = crossbar_min[index]

        crossbar_max_indices = np.random.choice(
            np.arange(torch.numel(crossbar_max)),
            replace=False,
            size=int(torch.numel(crossbar_max) * lrs_proportion),
        )
        for index in crossbar_max_indices:
            crossbar.conductance_matrix[index] = crossbar_max[index]

        crossbar.conductance_matrix = crossbar.conductance_matrix.view(crossbar_shape)
        return crossbar

    hrs_proportion = hrs_proportion + electroform_proportion
    for i in range(len(layer.crossbars)):
        layer.crossbars[i] = apply_device_faults_to_crossbar(
            layer.crossbars[i], lrs_proportion, hrs_proportion
        )

    return layer


def apply_cycle_variability(
    layer,
    distribution=torch.distributions.normal.Normal,
    min=0,
    max=float("Inf"),
    parallelize=False,
    r_off_kwargs={},
    r_on_kwargs={},
):
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
    r_off_kwargs : dict
        r_off kwargs.
    r_on_kwargs : dict
        r_on kwargs.
    """
    device = torch.device("cpu" if "cpu" in memtorch.__version__ else "cuda")

    def apply_cycle_variability_to_crossbar(
        crossbar,
        distribution=torch.distributions.normal.Normal,
        min=0,
        max=float("Inf"),
        parallelize=False,
        r_off_kwargs={},
        r_on_kwargs={},
    ):
        assert issubclass(
            distribution, torch.distributions.distribution.Distribution
        ), "Distribution is not in torch.distributions."
        for arg in inspect.signature(distribution).parameters.values():
            if (
                arg.name not in r_off_kwargs or arg.name not in r_on_kwargs
            ) and arg.name is not "validate_args":
                raise Exception(
                    "Argument %s is required for %s" % (arg.name, distribution)
                )

        shape = crossbar.conductance_matrix.shape
        r_off = distribution(**r_off_kwargs).sample(sample_shape=shape).clamp(min, max)
        r_on = distribution(**r_on_kwargs).sample(sample_shape=shape).clamp(min, max)
        if parallelize:

            def write_r_off(device, conductance):
                device.r_off = conductance

            def write_r_on(device, conductance):
                device.r_on = conductance

            np.frompyfunc(write_r_off, 2, 0)(crossbar.devices, r_off)
            np.frompyfunc(write_r_on, 2, 0)(crossbar.devices, r_on)
        else:
            if layer.tile_shape is not None:
                for i in range(0, layer.crossbars[0].devices.shape[0]):
                    for j in range(0, layer.crossbars[0].devices.shape[1]):
                        for k in range(0, layer.crossbars[0].devices.shape[2]):
                            crossbar.devices[i][j][k].r_off = r_off[i][j][k].item()
                            crossbar.devices[i][j][k].r_on = r_on[i][j][k].item()
            else:
                for i in range(0, crossbar.rows):
                    for j in range(0, crossbar.columns):
                        crossbar.devices[i][j].r_off = r_off[i][j].item()
                        crossbar.devices[i][j].r_on = r_on[i][j].item()

        crossbar.conductance_matrix = torch.max(
            torch.min(crossbar.conductance_matrix.clone().detach().cpu(), 1 / r_on),
            1 / r_off,
        ).to(device)
        crossbar.update(from_devices=False)
        return crossbar

    for i in range(len(layer.crossbars)):
        layer.crossbars[i] = apply_cycle_variability_to_crossbar(
            layer.crossbars[i],
            distribution,
            min=min,
            max=max,
            parallelize=parallelize,
            r_off_kwargs=r_off_kwargs,
            r_on_kwargs=r_on_kwargs,
        )

    return layer
