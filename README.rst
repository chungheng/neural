=============================================================
Neural: a simple python package for neuron and synapse models
=============================================================

.. image:: https://img.shields.io/pypi/v/neural.svg
        :target: https://pypi.python.org/pypi/neural

.. image:: https://img.shields.io/travis/TK-21st/neural.svg
        :target: https://travis-ci.com/TK-21st/neural

.. image:: https://readthedocs.org/projects/neural/badge/?version=latest
        :target: https://neural.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


Neural is a Python library consisting of implementation of neuron and synapse models with
support of various numerical solvers. Without using a specialized specification language
for neuron or synapse models, PyNeural defines a computational model as a set of ordinary
differential equations in plain Python syntax. For example, a model for computing parabola
trajectory can be expressed as

.. code-block:: python

    >>> class Parabola(Model):
    >>>     ...
    >>>    def ode(self):
    >>>        self.d_x = x


* Free software: BSD license
* Documentation: https://neural.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
