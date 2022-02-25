memtorch.bh
===========
Submodule containing various memristive device behavioral models and methods to simualte non-ideal device and circuit behavior.

memtorch.bh.memristor
---------------------
All memristor models and window functions are encapsulated and documented in :doc:`memtorch.bh.memristor <../memtorch.bh.memristor>`.

memtorch.bh.nonideality
-----------------------
All non-idealities modelled by MemTorch are encapsulated and documented in :doc:`memtorch.bh.nonideality <../memtorch.bh.nonideality>`.

memtorch.bh.crossbar.Crossbar
-----------------------------
Class used to model memristor crossbars and to manage modular crossbar tiles.

.. code-block:: python

  import torch
  import memtorch

  crossbar = memtorch.bh.crossbar.Crossbar(memtorch.bh.memristor.VTEAM,
                                           {"r_on": 1e2, "r_off": 1e4},
                                           shape=(100, 100),
                                           tile_shape=(64, 64))
  crossbar.write_conductance_matrix(torch.zeros(100, 100).uniform_(1e-2, 1e-4), transistor=True)
  crossbar.devices[0][0][0].set_conductance(1e-4)
  crossbar.update(from_devices=True, parallelize=True)

.. note::
  **use_bindings** is enabled by default, to accelerate operation using C++/CUDA (if supported) bindings.

.. warning::
  As of version 1.1.6, the **write_conductance_matrix** method exhibits different behavior when **self.use_bindings** is True, **CUDA** operation is enabled, and the **Data_Driven2021** memristor model is used.

  When **self.use_bindings** is True, **CUDA** operation is enabled, and the **Data_Driven2021** memristor model is used, the programming voltage is force adjusted by **force_adjustment_voltage** when a device becomes stuck.
  For all others models, or when **CUDA** operation is not enabled or **self.use_bindings** is false, the conductance state of the device being modelled is adjusted using **force_adjustment** when it becomes stuck.

  This behavior will made consistent across Python, C++, and CUDA bindings, in a future release.

.. automodule:: memtorch.bh.crossbar.Crossbar
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.crossbar.Program
----------------------------
Methods to program (alter) the conductance devices within a crossbar or modular crossbar tiles.

.. automodule:: memtorch.bh.crossbar.Program
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.crossbar.Tile
-------------------------

.. automodule:: memtorch.bh.crossbar.Tile
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.Quantize
--------------------
Wrapper for C++ quantization bindings.

.. automodule:: memtorch.bh.Quantize
   :members:
   :undoc-members:
   :show-inheritance:

memtorch.bh.StochasticParameter
-------------------------------
Methods to model stochastic parameters. 

**memtorch.bh.StochasticParameter** is most commonly used to define stochastic parameters when defining behavioural memristor models, as follows:

.. code-block:: python

  import torch
  import memtorch

  crossbar = memtorch.bh.crossbar.Crossbar(memtorch.bh.memristor.VTEAM,
                                           {"r_on": memtorch.bh.StochasticParameter(min=1e3, max=1e2), "r_off": 1e4},
                                           shape=(100, 100),
                                           tile_shape=(64, 64))

.. automodule:: memtorch.bh.StochasticParameter
   :members:
   :undoc-members:
   :show-inheritance:
