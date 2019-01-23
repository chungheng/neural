.. -*- rst -*-


PyNeural
========

PyNeural is a Python library consisting of implementation of neuron and synapse models with support of various numerical solvers. Without using a specialized specification language for neuron or synapse models, PyNeural defines a computational model as a set of ordinary differential equations in plain Python syntax. For example, a model for computing parabola trajectory can be expressed as

.. code-block:: python
    >>> class Parabola(Model):
    >>>     ...
    >>>    def ode(self):
    >>>        self.d_x = x



.. toctree::
   :maxdepth: 2
   :caption: Contents:



Index
==================

* :ref:`genindex`
