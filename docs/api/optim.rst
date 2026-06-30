Optimization Module
===================

The optimization module provides optimizers for parameter training.

.. autosummary::
   :toctree: generated
   :recursive:

   descent.optim

PyTorch Optimizers
------------------

DESCENT also supports all standard PyTorch optimizers:

- ``torch.optim.Adam``
- ``torch.optim.AdamW``
- ``torch.optim.SGD``
- ``torch.optim.LBFGS``

See the `PyTorch documentation <https://docs.pytorch.org/docs/stable/optim.html>`__ for details.

Learning Rate Schedules
-----------------------

DESCENT supports all PyTorch learning rate schedulers:

- ``torch.optim.lr_scheduler.StepLR``
- ``torch.optim.lr_scheduler.ReduceLROnPlateau``
- ``torch.optim.lr_scheduler.CosineAnnealingLR``
- ``torch.optim.lr_scheduler.ExponentialLR``

See the `PyTorch optimizer documentation <https://docs.pytorch.org/docs/stable/optim.html>`__ for details.
