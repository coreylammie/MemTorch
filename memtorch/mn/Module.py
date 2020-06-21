import memtorch
from memtorch.map.Parameter import naive_map
from .Linear import Linear
from .Conv1d import Conv1d
from .Conv2d import Conv2d
import torch
import torch.functional as F
import multiprocessing as mp
import itertools


supported_module_parameters = {'<class \'torch.nn.modules.linear.Linear\'>': Linear,
                           '<class \'torch.nn.modules.conv.Conv1d\'>': Conv1d,
                           '<class \'torch.nn.modules.conv.Conv2d\'>': Conv2d
                           }

def patch_model(model, memristor_model, memristor_model_params, module_parameters_to_patch={}, mapping_routine=naive_map, p_l=None, transistor=True, programming_routine=None, scheme=memtorch.bh.Scheme.DoubleColumn, **kwargs):
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
    p_l: float
        If not None, the proportion of weights to retain.
    transistor : bool
        Used to determine if a 1T1R (True) or 1R arrangement (False) is simulated.
    programming_routine : function
        Programming routine to use.
    scheme : memtorch.bh.Scheme
        Weight representation scheme.

    Returns
    -------
    torch.nn.Module
        Patched torch.nn.Module.
    """
    model.map = mapping_routine
    for i, (name, m) in enumerate(list(model.named_modules())):
        for parameter in module_parameters_to_patch:
            if isinstance(m, parameter):
                if 'cpu' not in memtorch.__version__ and len(name.split('.')) > 1:
                    name = name.split('.')[1]

                parameter_type = str(type(m))
                patch = supported_module_parameters.get(parameter_type)
                assert parameter_type in supported_module_parameters, 'Patching of %s is not currently supported' % type(m)
                if hasattr(model, 'module'):
                    setattr(model.module, name, patch(m,
                                                      memristor_model=memristor_model,
                                                      memristor_model_params=memristor_model_params,
                                                      mapping_routine=mapping_routine,
                                                      transistor=transistor,
                                                      programming_routine=programming_routine,
                                                      p_l=p_l,
                                                      scheme=scheme,
                                                      **kwargs))
                else:
                    setattr(model, name, patch(m,
                                                      memristor_model=memristor_model,
                                                      memristor_model_params=memristor_model_params,
                                                      mapping_routine=mapping_routine,
                                                      transistor=transistor,
                                                      programming_routine=programming_routine,
                                                      p_l=p_l,
                                                      scheme=scheme,
                                                      **kwargs))

    def tune_(self):
        """Method to tune a memristive layer."""
        for i, (name, m) in enumerate(list(self.named_modules())):
            if hasattr(m, 'tune'):
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
                delattr(m, 'weight')
                m.weight = None

        if 'cpu' not in memtorch.__version__:
            torch.cuda.empty_cache()
            
        self.forward_legacy(False)
        delattr(self, 'forward_legacy')

    model.forward_legacy = forward_legacy.__get__(model)
    model.tune_ = tune_.__get__(model)
    model.forward_legacy(False)
    model.disable_legacy = disable_legacy.__get__(model)
    return model
