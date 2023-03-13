.. SHARC documentation master file, created by
   sphinx-quickstart on Mon Mar  6 17:20:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SHARC's documentation!
=================================
**SHARC** (**SHA**\ rpened Dimensionality **R**\ eduction & **C**\ lassification) is a Python library for performing 
local gradient clustering (LGC) based sharpened dimensionality reduction (SDR) using neural network projections (NNP)
and constructing classifiers that use these projections to make classifications. The library also contains functions for finding
the optimal SDR parameters and for consolidating classification results obtained through multiple classifiers.

Installation
------------
Before installing **SHARC** make sure **pySDR** is installed first. To install **pySDR** please follow its
respective `installation instructions <https://martenlourens.github.io/pySDR/#installation>`_.

To install **SHARC** go inside the project directory (i.e. the directory containing the *setup.py* file) and run:

.. code-block:: bash

   $ pip install .

Contents
--------
.. toctree::
   :maxdepth: 2

   api
