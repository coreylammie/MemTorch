from typing import List, Optional

import torch
from torch.nn import Module
from torchinfo import summary

from utils.logger import logger
from utils.output import save_network_info


def network_metrics(
    network: Module,
    input_dim: List,
    device: Optional[torch.device],
    save_output: bool = True,
) -> dict:
    """
    Extract useful information from the network.

    :param network: The network to analyse
    :param input_dim: The dimension of the input
    :param device: The device use (cpu or cuda)
    :param save_output: If true the metrics will be saved in a text file in the run directory
    :return: A dictionary of metrics with their values
    """
    network_info = summary(network, input_size=input_dim, device=device, verbose=0)

    logger.debug("Network info:\n" + str(network_info))

    metrics = {
        "name": type(network).__name__,
        "loss_function": network.get_loss_name(),
        "optimizer_function": network.get_optimizer_name(),
        "device": str(device),
        "total_params": network_info.total_params,
        "trainable_params": network_info.trainable_params,
        "non_trainable_params": network_info.total_params
        - network_info.trainable_params,
        "MAC_operations": network_info.total_mult_adds,
        "input_dimension": input_dim,
    }

    if save_output:
        save_network_info(metrics)

    return metrics
