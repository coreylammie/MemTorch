memtorch.mn
===========
Memristive `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ equivalent submodule.

memtorch.mn.Module
------------------
Encapsulates :class:`memtorch.bmn.Module.patch_model`, which can be used to convert `torch.nn <https://pytorch.org/docs/stable/nn.html>`_ models. 

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

.. warning::
  It is strongly suggested to copy the original model using **copy.deepcopy** prior to conversion, as some values are overriden by-reference.

.. automodule:: memtorch.mn.Module
   :members:
   :undoc-members:
   :show-inheritance:

The following layer/module types are currently supported:

memtorch.mn.Linear
------------------
`torch.nn.Linear <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>`_ equivalent.

.. automodule:: memtorch.mn.Linear
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.mn.Conv1d
------------------
`torch.nn.Conv1d <https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html>`_ equivalent.

.. automodule:: memtorch.mn.Conv1d
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.mn.Conv2d
------------------
`torch.nn.Conv2d <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_ equivalent.

.. automodule:: memtorch.mn.Conv2d
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.mn.Conv3d
------------------
`torch.nn.Conv3d <https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html>`_ equivalent.

.. automodule:: memtorch.mn.Conv3d
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.mn.RNN
------------------
`torch.nn.RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_ equivalent.

.. automodule:: memtorch.mn.RNN
   :members:
   :undoc-members:
   :show-inheritance: