import itertools
import multiprocessing as mp

import torch
import torch.functional as F
from torch.nn import modules
from torch.nn.modules import module

import memtorch
from memtorch.map.Input import naive_scale
from memtorch.map.Parameter import naive_map

from .Conv1d import Conv1d
from .Conv2d import Conv2d
from .Conv3d import Conv3d
from .Linear import Linear

supported_module_parameters = {
    "<class 'torch.nn.modules.linear.Linear'>": Linear,
    "<class 'torch.nn.modules.conv.Conv1d'>": Conv1d,
    "<class 'torch.nn.modules.conv.Conv2d'>": Conv2d,
    "<class 'torch.nn.modules.conv.Conv3d'>": Conv3d,
}


def patch_model(
    model,
    memristor_model,
    memristor_model_params,
    module_parameters_to_patch={},
    mapping_routine=naive_map,
    transistor=True,
    programming_routine=None,
    programming_routine_params={"rel_tol": 0.1},
    p_l=None,
    scheme=memtorch.bh.Scheme.DoubleColumn,
    tile_shape=None,
    max_input_voltage=None,
    scaling_routine=naive_scale,
    scaling_routine_params={},
    source_resistance=None,
    line_resistance=None,
    ADC_resolution=None,
    ADC_overflow_rate=0.0,
    quant_method=None,
    use_bindings=True,
    random_crossbar_init=False,
    verbose=True,
    **kwargs
):
    """Method to convert a torch.nn model to a memristive model.

    Parameters
    ----------
    model : torch.nn.Module
        torch.nn.Module to patch.
    memristor_model : memtorch.bh.memristor.Memristor.Memristor
        Memristor model.
    memristor_model_params : **kwargs
        Memristor model keyword arguments.
    module_parameters_to_patch : module_paramter_patches
        Model parameters to patch.
    mapping_routine : function
        Mapping routine to use.
    transistor : bool
        Used to determine if a 1T1R (True) or 1R arrangement (False) is simulated.
    programming_routine : function
        Programming routine to use.
    programming_routine_params : **kwargs
        Programming routine keyword arguments.
    p_l: float
        If not None, the proportion of weights to retain.
    scheme : memtorch.bh.Scheme
        Weight representation scheme.
    tile_shape : (int, int)
        Tile shape to use to store weights. If None, modular tiles are not used.
    max_input_voltage : float
        Maximum input voltage used to encode inputs. If None, inputs are unbounded.
    scaling_routine : function
        Scaling routine to use in order to scale batch inputs.
    scaling_routine_params : **kwargs
        Scaling routine keyword arguments.
    source_resistance : float
        The resistance between word/bit line voltage sources and crossbar(s).
    line_resistance : float
        The interconnect line resistance between adjacent cells.
    ADC_resolution : int
        ADC resolution (bit width). If None, quantization noise is not accounted for.
    ADC_overflow_rate : float
        Overflow rate threshold for linear quanitzation (if ADC_resolution is not None).
    quant_method:
        Quantization method. Must be in ['linear', 'log', 'log_minmax', 'minmax', 'tanh'], or None.
    use_bindings : bool
        Used to determine if C++/CUDA bindings are used (True) or not (False).
    random_crossbar_init : bool
        Determines if the crossbar is to be initialized at random values in between Ron and Roff
    verbose : bool
        Used to determine if verbose output is enabled (True) or disabled (False).

    Returns
    -------
    torch.nn.Module
        Patched torch.nn.Module.
    """

    def patch_module(target_attr):
        parameter_type = str(type(target_attr))
        patch = supported_module_parameters.get(parameter_type)
        return patch(
            target_attr,
            memristor_model=memristor_model,
            memristor_model_params=memristor_model_params,
            mapping_routine=mapping_routine,
            transistor=transistor,
            programming_routine=programming_routine,
            programming_routine_params=programming_routine_params,
            p_l=p_l,
            scheme=scheme,
            tile_shape=tile_shape,
            max_input_voltage=max_input_voltage,
            scaling_routine=scaling_routine,
            scaling_routine_params=scaling_routine_params,
            source_resistance=source_resistance,
            line_resistance=line_resistance,
            ADC_resolution=ADC_resolution,
            ADC_overflow_rate=ADC_overflow_rate,
            quant_method=quant_method,
            use_bindings=use_bindings,
            random_crossbar_init=random_crossbar_init,
            verbose=verbose,
            **kwargs
        )

    def patch_modules(module, name=""):
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if any(
                isinstance(target_attr, module_parameter)
                and not hasattr(target_attr, "transistor")
                for module_parameter in module_parameters_to_patch
            ):
                new_bn = patch_module(target_attr)
                setattr(module, attr_str, new_bn)

        if isinstance(module, torch.nn.Module):
            if type(module) == torch.nn.modules.container.Sequential:
                for idx, (name, child) in enumerate(module.named_children()):
                    if any(
                        isinstance(child, module_parameter)
                        and not hasattr(child, "transistor")
                        for module_parameter in module_parameters_to_patch
                    ):
                        target_attr = module[idx]
                        new_bn = patch_module(target_attr)
                        module[idx] = new_bn
                    else:
                        patch_modules(child, name)
            else:
                for name, child in module.named_children():
                    patch_modules(child, name)
        else:
            for child in module:
                patch_modules(child, name)

    patch_modules(model)

    def tune_(self, tune_kwargs=None):
        """Method to tune a memristive layer.
        Parameters
        ----------
        tune_kwargs : dict
            Dictionary of **kwargs for different layer types for .tune().
        """
        for _, (name, m) in enumerate(list(self.named_modules())):
            if hasattr(m, "tune"):
                if tune_kwargs is not None:
                    module_type = str(type(m))
                    if module_type in tune_kwargs:
                        m.tune(**tune_kwargs[module_type])
                    else:
                        m.tune()
                else:
                    m.tune()

    def forward_legacy(self, enable_forward_legacy):
        """Method to enable or disable forward legacy operation.

        Parameters
        ----------
        enable_forward_legacy : bool
            Enable or disable forward legacy operation.
        """
        for i, (name, m) in enumerate(list(self.named_modules())):
            if type(m) in supported_module_parameters.values():
                m.forward_legacy_enabled = enable_forward_legacy

    def disable_legacy(self):
        """Method to delete all legacy parameters to reduce memory usage. When this method is called forward_legacy is disabled."""
        for i, (name, m) in enumerate(list(self.named_modules())):
            if type(m) in supported_module_parameters.values():
                delattr(m, "weight")
                m.weight = None

        if "cpu" not in memtorch.__version__:
            torch.cuda.empty_cache()

        self.forward_legacy(False)
        delattr(self, "forward_legacy")

    def set_cuda_malloc_heap_size(self, cuda_malloc_heap_size):
        """Method to set the CUDA malloc heap size."""
        for i, (name, m) in enumerate(list(self.named_modules())):
            if type(m) in supported_module_parameters.values():
                m.cuda_malloc_heap_size = cuda_malloc_heap_size

    model.forward_legacy = forward_legacy.__get__(model)
    model.tune_ = tune_.__get__(model)
    model.forward_legacy(False)
    model.disable_legacy = disable_legacy.__get__(model)
    model.set_cuda_malloc_heap_size = set_cuda_malloc_heap_size.__get__(model)
    return model
