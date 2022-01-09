memtorch.bh.nonideality
=======================
Submodule containing various models, which can be used to introduce various non-ideal device characteristics using :class:`memtorch.bh.nonideality.NonIdeality.apply_nonidealities`.

memtorch.bh.nonideality.NonIdeality
-----------------------------------
Class used to introduce/model non-ideal device and circuit characteristics. :class:`patched_model.apply_nonidealities` is commonly used to introduce such characteristics, as demonstrated by the following example:

.. code-block:: python

  import copy
  import Net
  from memtorch.mn.Module import patch_model
  from memtorch.map.Parameter import naive_map
  from memtorch.map.Input import naive_scale

  model = Net()
  reference_memristor = memtorch.bh.memristor.VTEAM
  patched_model = patch_model(copy.deepcopy(model),
                        memristor_model=reference_memristor,
                        memristor_model_params={},
                        module_parameters_to_patch=[torch.nn.Linear, torch.nn.Conv2d],
                        mapping_routine=naive_map,
                        transistor=True,
                        programming_routine=None,
                        tile_shape=(128, 128),
                        max_input_voltage=0.3,
                        scaling_routine=naive_scale,
                        ADC_resolution=8,
                        ADC_overflow_rate=0.,
                        quant_method='linear')
   # Example usage of memtorch.bh.nonideality.NonIdeality.DeviceFaults
   patched_model = patched_model.apply_nonidealities(patched_model,
                                                     non_idealities=[memtorch.bh.nonideality.NonIdeality.DeviceFaults],
                                                     lrs_proportion=0.25,
                                                     hrs_proportion=0.10,
                                                     electroform_proportion=0)

.. automodule:: memtorch.bh.nonideality.NonIdeality
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.nonideality.FiniteConductanceStates
-----------------------------------------------

.. automodule:: memtorch.bh.nonideality.FiniteConductanceStates
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.nonideality.DeviceFaults
------------------------------------
Methods used to model device faults.

.. automodule:: memtorch.bh.nonideality.DeviceFaults
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.nonideality.NonLinear
---------------------------------

.. automodule:: memtorch.bh.nonideality.NonLinear
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.nonideality.Endurance
---------------------------------

.. automodule:: memtorch.bh.nonideality.Endurance
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.nonideality.Retention
---------------------------------

.. automodule:: memtorch.bh.nonideality.Retention
   :members:
   :undoc-members:
   :show-inheritance:

For both :class:`memtorch.bh.nonideality.Endurance` and :class:`memtorch.bh.nonideality.Retention`, the following internal endurance and retention models are natively supported:

memtorch.bh.nonideality.endurance_retention_models.conductance_drift
--------------------------------------------------------------------
.. automodule:: memtorch.bh.nonideality.endurance_retention_models.conductance_drift
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM
-----------------------------------------------------------------------------
.. automodule:: memtorch.bh.nonideality.endurance_retention_models.empirical_metal_oxide_RRAM
   :members:
   :undoc-members:
   :show-inheritance:
   