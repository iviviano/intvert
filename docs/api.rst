API Reference
=============

The package is separated into three groups of procedures. The :ref:`Fourier Transforms <ft>` perform 1D and 2D discrete Fourier transforms (DFT's) with mixed precision floating point operations. For DFT conventions in 1D and 2D, see `mp_dft` and `mp_dft2`, respectively. The :ref:`Sampling <sample>` partition DFT frequencies and sample signals at partial sets of frequencies. The :ref:`Inversion <invert>` procedures invert sampled integer signals. The inversion algorithms implement the efficient dynammic programming algorithms in [LV]_.

.. currentmodule:: intvert

.. _ft:

Fourier Transforms
------------------

.. autosummary::
	:toctree: _autosummary_functions

	mp_dft
	mp_idft
	mp_dft2
	mp_idft2

.. _sample:

Sampling
--------

.. autosummary::
	:toctree: _autosummary_functions

	get_coeff_classes_1D
	get_coeff_classes_2D
	select_coeffs_1D
	select_coeffs_2D
	sample_1D
	sample_2D

.. _invert:

Inversion
---------

.. autosummary::
	:toctree: _autosummary_functions

	invert_1D
	invert_2D
	InversionError


Examples
--------
For examples, see the various functions.