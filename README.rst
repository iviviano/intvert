=======
intvert
=======

intvert is a pure Python package for inversion of 1D and 2D integer arrays from partial DFT samples. This package contains the codebase for the paper [LV]_. See the full documentation `here <https://intvert.readthedocs.io/en/latest/index.html>`_.

Examples
--------

An example usage of the sampling and inversion procedures in 2D for a large binary matrix.

>>> import intvert
>>> import numpy as np
>>> import gmpy2 
>>> 
>>> gen = np.random.default_rng(0)
>>> signal = gen.integers(0, 2, (30, 40)) # generate random binary matrix signal
>>> signal
array([[1, 1, 1, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 1, 0],
       [1, 1, 0, ..., 0, 1, 0],
       ...,
       [0, 1, 0, ..., 1, 1, 1],
       [1, 1, 1, ..., 0, 0, 1],
       [1, 1, 1, ..., 1, 1, 0]], shape=(30, 40)) 
>>> with gmpy2.get_context() as c: # perform sampling and inversion with increased precision
...     c.precision = 100
...     sampled = intvert.sample_2D(signal)
...     inverted = intvert.invert_2D(signal, beta2=1e20)
... 
>>> inverted
array([[1, 1, 1, ..., 0, 0, 0],
       [0, 0, 0, ..., 1, 1, 0],
       [1, 1, 0, ..., 0, 1, 0],
       ...,
       [0, 1, 0, ..., 1, 1, 1],
       [1, 1, 1, ..., 0, 0, 1],
       [1, 1, 1, ..., 1, 1, 0]], shape=(30, 40))
>>> np.allclose(signal, inverted) # inverted signal matches signal
True

Installation
------------

intvert may be installed from `PyPI <https://pypi.org/project/intvert/>`_ with `pip <https://pypi.org/project/pip/>`_.

.. code-block:: bash

       pip install intvert


References
----------
.. [LLL]

       Arjen K. Lenstra, Hendrik W. Lenstra, and László Lovász. Factoring polynomials with rational coefficients. Mathematische Annalen, 261(4):515–534, December 1982. ISSN 1432-1807. doi: 10.1007/bf01457454. URL: `<http://dx.doi.org/10.1007/BF01457454>`_.

.. [LV] 
       
       Howard W. Levinson and Isaac Viviano. Recovery of integer images from limited dft measurements with lattice methods, 2025. URL: `<https://arxiv.org/abs/2510.11949>`_.

.. [PC] 
       
       Soo-Chang Pei and Kuo-Wei Chang. Binary signal perfect recovery from partial dft coefficients. IEEE Transactions on Signal Processing, 70:3848–3861, 2022. doi: 10.1109/TSP.2022.3190615. URL: `<http://dx.doi.org/10.1109/TSP.2022.3190615>`_.


Requirements
------------
intvert relies on the following Python packages:
 - `numpy <https://numpy.org/doc/stable/>`_ for fast array operations
 - `gmpy2 <https://gmpy2.readthedocs.io/en/stable/>`_ for multiple precision floating point operations
 - `fpylll <https://fpylll.readthedocs.io/en/stable/>`_ for implementations of the LLL lattice basis reduction algorithm [LLL]_
 - `sympy <https://docs.sympy.org/latest/index.html>`_ for integer factorization
