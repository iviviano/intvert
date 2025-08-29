import fpylll as lll
import numpy as np
import gmpy2 as mp
import sympy as sp
from itertools import product

from .sample import *

_lll_params = {f"beta{n}" for n in range(4)} | {"delta", "epsilon"}

def _approximate_svp(basis, delta, fp_precision = None):
    
    precision = fp_precision if fp_precision else max(int(1.5 * mp.get_context().precision), 53)
    
    integer_basis = lll.IntegerMatrix.from_matrix(basis)
    with lll.FPLLL.precision(precision):
        lll.LLL.Reduction(lll.GSO.Mat(integer_basis), delta)()
    return np.array([basis_vec for basis_vec in integer_basis], dtype = object)


def _get_basis_matrix(signal, dft, inverted, known_coeffs, factors, use_guess=True):

	N = len(signal)

	top_block = np.concatenate([np.eye(N), -signal.reshape(N, 1) * use_guess], axis=1)

	penalty_block = [[0] * N + [1]]

	sums_block = np.concatenate([np.concatenate([np.eye(N // n)] * n + [-np.transpose([inverted[N//n]])], axis=1) for n in factors])

	if not known_coeffs:
		for ind in range(N):
			if np.gcd(ind, N) == 1 and abs(dft[ind]) > 1e-10:
				break
		known_coeffs = [ind]
	# root = mp.root_of_unity(N, N - 1)
	# coeff_block = [np.concatenate([root ** (np.arange(N) * ind), [-dft[ind]]]) for ind in known_coeffs]
	coeff_block = [np.concatenate([[mp.root_of_unity(N, int((N - 1) * k * ind % N)) for k in range(N)] + [-dft[ind]]]) for ind in known_coeffs]

	return np.concatenate([top_block, penalty_block, sums_block, mp_real(coeff_block), mp_imag(coeff_block)])


def _setup_and_solve(dft, inverted, known_coeffs, factors, use_guess=True, beta0=1e-1, beta1=1e3, beta2=1e14, beta3=1e2, delta=.9972, epsilon=None):

	signal = mp_real(mp_idft(dft))
	N = len(signal)
	if N == 1:
		return signal
	M = max(len(known_coeffs), 1)
	basis_matrix = _get_basis_matrix(signal, dft, known_coeffs=known_coeffs, inverted=inverted, factors=factors, use_guess=use_guess)
	
	precision = mp.get_context().precision
	if epsilon is None:
		epsilon = 10 ** (-.3 * precision + 2.7 + .1 * N)
	def check(vector):
		if len(vector) == N or vector[N] == beta0:
			return np.allclose(basis_matrix[N + 1:] @ np.concatenate([mp_round(vector[:N] + signal), [1]]), 0, atol=epsilon)

	if check(np.zeros(N)):
		# print(f"limited recon was correct (n = {N})")
		return mp_round(signal)

	scaled_basis_matrix = np.copy(basis_matrix)
	scaled_basis_matrix[N] *= beta0
	scaled_basis_matrix[N + 1: -2 * M] *= beta1
	scaled_basis_matrix[-2 * M:] *= beta2
	scaled_basis_matrix *= beta3

	reduced_basis = _approximate_svp(np.vectorize(int, otypes = [object])(scaled_basis_matrix).transpose(), delta=delta) / beta3

	for vector in reduced_basis:
		for sign in [-1, 1]:
			if check(sign * vector):
				return mp_round(sign * vector[:N] + signal)
	
	params = {'epsilon': epsilon, 'precision': precision, 'beta2': beta2}
	raise InversionError(f"Failure to recover a length {N} subproblem. It's possible that recovery was correct and the tolerance was too low. If you believer this is the case, try increasing epsilon. If recovery was incorrect, increasing precision and beta2 may aid in recovery.", **params)

class InversionError(Exception):

	"""Python-exception-derived object raised by inversion functions.

	This exception is raised by `invert_1D` and `invert_2D` when the they fail to solve any subproblem to within the given tolerance. Contains current values of relevant parameters at the time of error.

	Parameters
	----------
	msg: str
		The error message.
	beta2: float
		The value of the lattice parameter beta2.
	precision: int
		The number of bits of precision in the current `gmpy2` context.
	epsilon: float
		The value of the tolerance parameter epsilon.
	"""

	def __init__(self, msg, beta2, precision, epsilon):
		self.msg = msg
		self.beta2 = beta2
		self.precision = precision
		self.epsilon = epsilon
		super().__init__(msg)
	
	def __str__(self):
		return (self.msg
		 + " Current Parameters: \n"
		 + f"\tbeta2:     {self.beta2:.2e}\n"
		 + f"\tprecision: {self.precision}\n"
		 + f"\tepsilon:   {self.epsilon:.2e}"
		)
		

@np.vectorize(signature="(N)->(N)", excluded=set(range(1, len(_lll_params) + 1)) | {"known_coeffs"} | _lll_params)	
def invert_1D(signal, known_coeffs={}, **lattice_params):
	"""Invert an integer signal from limited DFT spectrum.

	Invert the last axis of an integer signal from a limited set of sampled DFT coefficients. The sampled frequencies may be provided in `known_coeffs`, which should be structured like the output of `select_coeffs_1D`. The input `signal` should be given in real space, so the known DFT coefficients are obtained from `mp_dft(signal)`. If no known frequencies are provided, they are chosen automatically from `signal` by assuming nonzero DFT coefficients are known. It is assumed that a sufficient set of coefficients are known to guarantee uniqueness of inversion. 

	Parameters
	----------
	signal : mpc ndarray
		Sampled signal.
	known_coeffs : dict, optional
		Dictionary of known frequencies, by default {}

	Returns
	-------
	int ndarray
		Inverted signal.

	Raises
	------
	InversionError
		If inversion fails for any subproblem. The current lattice parameter values are given, so they may be tuned to allow inversion.

	Other Parameters
	------------------
	These parameters are passed as keyword arguments through `**lattice_params`. They control the lattice-based integer programming solver.
	beta0 : float
		Penalty for coefficient of last lattice basis column, by default 1e-1
	beta1 : float
		Penalty for missing linear constraints with integer coefficients, by default 1e3
	beta2 : float
		Penalty for missing linear constraints with real coefficients, by default 1e14
	beta3 : float
		Rescale before truncation, by default 1e2
	delta : float
		LLL approximation parameter delta, by default 0.9972
	epsilon : float
		Absolute tolerance for verifying shortest vectors against DFT coefficient data.

	Notes
	-----
	This dynamic programming implementation of 1D inversion iterates through the divisors :math:`d` of the signal size `N = len(signal)`. Each iteration requires solving a linear integer program in :math:`d` variables. The integer program is reduced to the shortest vector problem by constructing the lattice, a lattice basis, with reduction parameters :math:`\beta_0,\beta_1,\beta_2`. This shortest vector problem is solved with the LLL approximation algorithm using the given value of :math:`\delta`. The vector returned by LLL is rejected if the known part of its DFT does not match `signal` to absolute tolerance `epsilon`, causing an `InversionError`.
	"""

	N = len(signal)
	dft = mp_dft(signal)
	inverted = {1: mp_real(dft[:1])}
	for d in sp.divisors(N)[1:]:

		current_coeffs = [k * d // N for k in known_coeffs[N // d]] if d in known_coeffs else []
		factors = sp.primefactors(d)

		inverted[d] = _setup_and_solve(dft[:N:N // d], inverted=inverted, known_coeffs=current_coeffs, factors=factors, **lattice_params)

	return inverted[N].astype(int)
"""
description:
This dynamic programming implementation of 2D inversion iterates through pairs of divisors of `N1, N2 = signal.shape`, with a 1D inversion occuring at each iteration.
"""
@np.vectorize(signature="(M,N)->(M,N)", excluded=set(range(1, len(_lll_params) + 1)) | {"known_coeffs"} | _lll_params)	
def invert_2D(signal, known_coeffs={}, **lattice_params):

	M, N = signal.shape
	dft = mp_dft2(signal)
	dsums = {}
	
	all_coeff_classes = get_coeff_classes_2D(M, N, False)

	for N1, N2 in product(sp.divisors(M), sp.divisors(N)):

		len = np.lcm(N1, N2)
		factors = sp.primefactors(len)
		lams = np.arange(len)

		for class_ in all_coeff_classes[M // N1, N // N2]:

			for coeffs in known_coeffs[M // N1, N // N2] if (M // N1, N // N2) in known_coeffs else {}:
				if class_ & coeffs: break
			else:
				coeffs = []

			k, l = set(class_).pop()

			k1, l1 = k * lams % M, l * lams % N
			
			inverted = {}
			for p in factors:
				
				kp, lp = k1[p % len], l1[p % len]

				inverted[len // p] = dsums[kp, lp]

			direction_dft = dft[k1, l1]

			lll_result = _setup_and_solve(direction_dft, inverted=inverted, known_coeffs=[lam for lam in lams if (k1[lam], l1[lam]) in coeffs], factors=factors, **lattice_params)

			permutations = {} # should try to eliminate this
			for lam in lams:
				if np.gcd(lam, len) == 1:
					lam_inv = pow(int(lam), -1, int(len))
					permutations[k1[lam], l1[lam]] = lll_result[lam_inv * np.arange(len) % len]

			dsums.update(permutations)
			# dsums[N1 % M, N2 % N][k, l] = lll_result
			
			dft[k1, l1] = mp_dft(lll_result)

	return mp_round(mp_real(mp_idft2(dft))).astype(int)


if __name__ == "__main__":

	np.random.seed(3)

	M, N = 7, 7

	signal = np.random.randint(0, 2, M)
	blurred = sample_1D(signal)
	inverted = invert_1D(blurred)
	assert np.allclose(signal - inverted, 0), f"actual: {inverted}, expected: {signal}"
	print("1D reconstruction correct")

	signal = np.random.randint(0, 2, (M, N))
	blurred = sample_2D(signal)
	inverted = invert_2D(blurred)
	assert np.allclose(signal - inverted, 0)
	print("2D reconstruction correct")