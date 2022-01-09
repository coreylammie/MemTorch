memtorch.map
============
Submodule containing various mapping, scaling, and encoding methods.

memtorch.map.Input
-------------------
Encapsulates internal methods to encode (scale) input values as bit-line voltages. Methods can either be specified when converting individual layers:

.. code-block:: python

  from memtorch.map.Input import naive_scale

  m = memtorch.mn.Linear(torch.nn.Linear(10, 10), 
                         memtorch.bh.memristor.VTEAM, 
                         {}, 
                         tile_shape=(64, 64),
                         scaling_routine=naive_scale)

or when converting :class:`torch.nn.Module` instances:

.. code-block:: python

  import copy
  from memtorch.mn.Module import patch_model
  from memtorch.map.Input import naive_scale
  import Net

  model = Net()
  patched_model = patch_model(copy.deepcopy(model),
                              memtorch.bh.memristor.VTEAM,
                              {},
                              module_parameters_to_patch=[torch.nn.Linear],
                              scaling_routine=naive_scale)

.. automodule:: memtorch.map.Input
   :members:
   :undoc-members:
   :show-inheritance:

.. note::
  **force_scale** is used to specify whether inputs smaller than or equal to **max_input_voltage** are scaled or not.

memtorch.map.Module
-------------------
Encapsulates internal methods to determine relationships between readout currents of memristive crossbars and desired outputs.

.. warning::
  Currently, only **naive_tune** is supported. In a future release, externally-defined methods will be supported. 




.. automodule:: memtorch.map.Module
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.map.Parameter
----------------------
Encapsulates internal methods to naively map network parameters to memristive device conductance values. Methods can either be specified when converting individual layers:

.. code-block:: python

  from memtorch.map.Parameter import naive_map

  m = memtorch.mn.Linear(torch.nn.Linear(10, 10), 
                         memtorch.bh.memristor.VTEAM, 
                         {}, 
                         tile_shape=(64, 64),
                         mapping_routine=naive_map)

or when converting :class:`torch.nn.Module` instances:

.. code-block:: python

  import copy
  from memtorch.mn.Module import patch_model
  from memtorch.map.Parameter import naive_map
  import Net

  model = Net()
  patched_model = patch_model(copy.deepcopy(model),
                              memtorch.bh.memristor.VTEAM,
                              {},
                              module_parameters_to_patch=[torch.nn.Linear],
                              mapping_routine=naive_map)

.. automodule:: memtorch.map.Parameter
   :members:
   :undoc-members:
   :show-inheritance:
