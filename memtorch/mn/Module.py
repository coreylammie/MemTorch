import itertools
import multiprocessing as mp

import torch
import torch.functional as F

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
    ADC_resolution=None,
    ADC_overflow_rate=0.0,
    quant_method=None,
    use_bindings=True,
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
    ADC_resolution : int
        ADC resolution (bit width). If None, quantization noise is not accounted for.
    ADC_overflow_rate : float
        Overflow rate threshold for linear quanitzation (if ADC_resolution is not None).
    quant_method:
        Quantization method. Must be in ['linear', 'log', 'log_minmax', 'minmax', 'tanh'], or None.
    use_bindings : bool
        Used to determine if C++/CUDA bindings are used (True) or not (False).
    verbose : bool
        Used to determine if verbose output is enabled (True) or disabled (False).

    Returns
    -------
    torch.nn.Module
        Patched torch.nn.Module.
    """
    model.map = mapping_routine
    for i, (name, m) in enumerate(list(model.named_modules())):
        for parameter in module_parameters_to_patch:
            if isinstance(m, parameter):
                if "cpu" not in memtorch.__version__ and len(name.split(".")) > 1:
                    name = name.split(".")[1]

                parameter_type = str(type(m))
                patch = supported_module_parameters.get(parameter_type)
                assert (
                    parameter_type in supported_module_parameters
                ), "Patching of %s is not currently supported" % type(m)
                if hasattr(model, "module"):
                    setattr(
                        model.module,
                        name,
                        patch(
                            m,
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
                            ADC_resolution=ADC_resolution,
                            ADC_overflow_rate=ADC_overflow_rate,
                            quant_method=quant_method,
                            use_bindings=use_bindings,
                            verbose=verbose,
                            **kwargs
                        ),
                    )
                else:
                    setattr(
                        model,
                        name,
                        patch(
                            m,
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
                            ADC_resolution=ADC_resolution,
                            ADC_overflow_rate=ADC_overflow_rate,
                            quant_method=quant_method,
                            use_bindings=use_bindings,
                            verbose=verbose,
                            **kwargs
                        ),
                    )

    def tune_(self, tune_kwargs=None):
        """Method to tune a memristive layer.

        Parameters
        ----------
        tune_kwargs : dict
            Dictionary of **kwargs for different layer types for .tune().
        """
        for i, (name, m) in enumerate(list(self.named_modules())):
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

    model.forward_legacy = forward_legacy.__get__(model)
    model.tune_ = tune_.__get__(model)
    model.forward_legacy(False)
    model.disable_legacy = disable_legacy.__get__(model)
    return model
