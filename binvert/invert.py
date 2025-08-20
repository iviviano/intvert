import fpylll as lll
import numpy as np
import gmpy2 as mp
import sympy as sp
from itertools import product

from .blur import *

_lll_params = {f"beta{n}" for n in range(4)} | {"delta"}

def approximate_svp(basis, delta, fp_precision = None):
    
    precision = fp_precision if fp_precision else max(int(1.5 * mp.get_context().precision), 53)
    
    integer_basis = lll.IntegerMatrix.from_matrix(basis)
    with lll.FPLLL.precision(precision):
        lll.LLL.Reduction(lll.GSO.Mat(integer_basis), delta)()
    return np.array([basis_vec for basis_vec in integer_basis], dtype = object)


def get_basis_matrix(signal, dft, inverted, known_coeffs, factors, use_guess=True):

	N = len(signal)

	top_block = np.concatenate([np.eye(N), -signal.reshape(N, 1) * use_guess], axis=1)

	penalty_block = [[0] * N + [1]]

	sums_block = np.concatenate([np.concatenate([np.eye(N // n)] * n + [-np.transpose([inverted[N//n]])], axis=1) for n in factors])

	root = mp.root_of_unity(N, N - 1)
	if not known_coeffs:
		for ind in range(N):
			if np.gcd(ind, N) == 1 and abs(dft[ind]) > 1e-10:
				break
		known_coeffs = [ind]
	coeff_block = [np.concatenate([root ** (np.arange(N) * ind), [-dft[ind]]]) for ind in known_coeffs]

	return np.concatenate([top_block, penalty_block, sums_block, mp_real(coeff_block), mp_imag(coeff_block)])


def setup_and_solve(dft, inverted, known_coeffs, factors, use_guess=True, beta0=1e-1, beta1=1e2, beta2=1e10, beta3=1e2, delta=.9972):

	signal = mp_real(mp_idft(dft))
	N = len(signal)
	if N == 1:
		return signal
	M = max(len(known_coeffs), 1)
	basis_matrix = get_basis_matrix(signal, dft, known_coeffs=known_coeffs, inverted=inverted, factors=factors, use_guess=use_guess)
	
	def check(vector):
		if len(vector) == N or vector[N] == beta0:
			return np.allclose(basis_matrix[N + 1:] @ np.concatenate([mp_round(vector[:N] + signal), [1]]), 0)

	if check(np.zeros(N)):
		# print(f"limited recon was correct (n = {N})")
		return mp_round(signal)

	scaled_basis_matrix = np.copy(basis_matrix)
	scaled_basis_matrix[N] *= beta0
	scaled_basis_matrix[N + 1: -2 * M] *= beta1
	scaled_basis_matrix[-2 * M:] *= beta2
	scaled_basis_matrix *= beta3

	reduced_basis = approximate_svp(np.vectorize(int, otypes = [object])(scaled_basis_matrix).transpose(), delta=delta) / beta3

	for vector in reduced_basis:
		for sign in [-1, 1]:
			if check(sign * vector):
				return mp_round(sign * vector[:N] + signal)
	
	raise Exception("failure")

@np.vectorize(signature="(N)->(N)", excluded=set(range(1, 6)) | {"known_coeffs"} | _lll_params)	
def invert_1D(signal, known_coeffs={}, **lll_params):

	N = len(signal)
	dft = mp_dft(signal)
	inverted = {1: mp_real(dft[:1])}
	for d in sp.divisors(N)[1:]:

		current_coeffs = [k * d // N for k in known_coeffs[N // d]] if d in known_coeffs else []
		factors = sp.primefactors(d)

		inverted[d] = setup_and_solve(dft[:N:N // d], inverted=inverted, known_coeffs=current_coeffs, factors=factors, **lll_params)

	return inverted[N]

@np.vectorize(signature="(M,N)->(M,N)", excluded=set(range(1, 6)) | {"known_coeffs"} | _lll_params)	
def invert_2D(signal, known_coeffs={}, **lll_params):

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

			lll_result = setup_and_solve(direction_dft, inverted=inverted, known_coeffs=[lam for lam in lams if (k1[lam], l1[lam]) in coeffs], factors=factors, **lll_params)

			permutations = {} # should try to eliminate this
			for lam in lams:
				if np.gcd(lam, len) == 1:
					lam_inv = pow(int(lam), -1, int(len))
					permutations[k1[lam], l1[lam]] = lll_result[lam_inv * np.arange(len) % len]

			dsums.update(permutations)
			# dsums[N1 % M, N2 % N][k, l] = lll_result
			
			dft[k1, l1] = mp_dft(lll_result)

	return mp_round(mp_real(mp_idft2(dft)))


if __name__ == "__main__":

	np.random.seed(3)

	M, N = 7, 7

	signal = np.random.randint(0, 2, M)
	blurred = blur_1D(signal)
	inverted = invert_1D(blurred)
	assert np.allclose(signal - inverted, 0), f"actual: {inverted}, expected: {signal}"
	print("1D reconstruction correct")

	signal = np.random.randint(0, 2, (M, N))
	blurred = blur_2D(signal)
	inverted = invert_2D(blurred)
	assert np.allclose(signal - inverted, 0)
	print("2D reconstruction correct")