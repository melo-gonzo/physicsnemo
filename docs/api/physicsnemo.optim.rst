PhysicsNeMo Optim
=================

.. automodule:: physicsnemo.optim
.. currentmodule:: physicsnemo.optim

The PhysicsNeMo Optim module provides optimization utilities for training physics-informed
machine learning models. These utilities are designed to work seamlessly with PyTorch's
optimizer ecosystem while providing additional functionality for complex training scenarios.

CombinedOptimizer
-----------------

The :class:`CombinedOptimizer` allows combining multiple PyTorch optimizers into a unified
interface. This is particularly useful when different parts of a model require different
optimization strategies - for example, using Adam for encoder layers and SGD with momentum
for decoder layers.

.. autoclass:: physicsnemo.optim.CombinedOptimizer
    :members:
    :show-inheritance:
    :special-members: __init__


LookSAM (experimental)
----------------------

:class:`~physicsnemo.experimental.optim.LookSAM` wraps a PyTorch optimizer with
sharpness-aware minimization that computes the extra ascent gradient only every
``k`` steps. :class:`~physicsnemo.experimental.optim.LookLayerSAM` uses
layer-wise scaled ascent steps. Both live in the experimental namespace, so
their API may change.

.. autoclass:: physicsnemo.experimental.optim.LookSAM
    :members: step, state_dict, load_state_dict, add_param_group, global_step, sam_step
    :show-inheritance:

.. autoclass:: physicsnemo.experimental.optim.LookLayerSAM
    :show-inheritance:
