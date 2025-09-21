=======
intvert
=======

intvert is a pure Python package for inversion of 1D and 2D integer arrays from partial DFT samples. This package contains the codebase for the paper [LV]_. 

Examples
--------
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

References
----------
.. [LV] TODO

Requirements
------------
``intvert`` relies on the following Python packages:
 - `numpy <https://numpy.org/doc/stable/>`_ for fast array operations
 - `gmpy2 <https://gmpy2.readthedocs.io/en/stable/>`_ for multiple precision floating point operations
 - `fpylll <https://fpylll.readthedocs.io/en/stable/>`_ for implementations of the LLL lattice basis reduction algorithm
 - `sympy <https://docs.sympy.org/latest/index.html>`_ for integer factorization
