import fpylll as lll
import numpy as np
import gmpy2 as mp
import sympy as sp
from itertools import product

from .sample import *

_lll_params = {f"beta{n}" for n in range(4)} | {"delta", "epsilon"}

def _approximate_svp(basis, delta, fp_precision = None):
    """Apply the LLL algorithm to reduce the lattice basis.

    Parameters
    ----------
    basis : int ndarray
        Lattice basis matrix with basis vectors stored as rows.
    delta : float
        LLL reduction parameter.
    fp_precision : int, optional
        Number of bits of precision to use in LLL floating point operations. Set automatically to 1.5 * the current gmpy2 precision if None, by default None

    Returns
    -------
    int ndarray
        LLL reduced basis
    """
    
    precision = fp_precision if fp_precision else max(int(1.5 * mp.get_context().precision), 53)
    
    integer_basis = lll.IntegerMatrix.from_matrix(basis)
    with lll.FPLLL.precision(precision):
        lll.LLL.Reduction(lll.GSO.Mat(integer_basis), delta)()
    return np.array([basis_vec for basis_vec in integer_basis], dtype = object)


def _get_basis_matrix(signal, dft, inverted, known_coeffs, factors):
    """Set up the lattice basis for inversion of a length `N` signal (for reduction of ILP to SVP).

    Parameters
    ----------
    signal : mpfr array
        1D array containing the sampled signal, which is used as the guess.
    dft : mpc array
        1D array containing the DFT of the signal.
    inverted : Dict
        Dictionary mapping prime divisors `p` of `N` to the solved subproblems of size `N/p`.
    known_coeffs : List
        List of frequencies that are known. If this is empty, they are chosen automatically by assuming non-zero DFT coefficients are known.
    factors : List
        List of the prime factors of `N`.

    Returns
    -------
    mpfr ndarray
        Lattice basis with basis vectors stored as columns.
    """

    N = len(signal)

    top_block = np.concatenate([np.eye(N), -signal.reshape(N, 1)], axis=1)

    penalty_block = [[0] * N + [1]]

    sums_block = np.concatenate([np.concatenate([np.eye(N // n)] * n + [-np.transpose([inverted[N//n]])], axis=1) for n in factors])

    if not known_coeffs:
        for ind in range(N):
            if np.gcd(ind, N) == 1 and abs(dft[ind]) > 1e-10:
                break
        known_coeffs = [ind]
    coeff_block = [np.concatenate([[mp.root_of_unity(N, int((N - 1) * k * ind % N)) for k in range(N)] + [-dft[ind]]]) for ind in known_coeffs]

    return np.concatenate([top_block, penalty_block, sums_block, mp_real(coeff_block), mp_imag(coeff_block)])


def _setup_and_solve(dft, inverted, known_coeffs, factors, beta0=1e-1, beta1=1e3, beta2=1e14, beta3=1e2, delta=.9972, epsilon=None):
    """Sets up the reduction of ILP to SVP and applies the LLL algorithm to approximate the shortest vector.

    Parameters
    ----------
    dft : mpc array
        Sampled DFT of the 1D signal to invert.
    inverted : Dict
        Dictionary mapping prime divisors `p` of `N` to the solved subproblems of size `N/p`.
    known_coeffs : List
        List of frequencies that are known. If this is empty, they are chosen automatically by assuming non-zero DFT coefficients are known.
    factors : List
        List of the prime factors of `N`.

    Returns
    -------
    mpfr ndarray
        The inverted 1D signal.

    Raises
    ------
    InversionError
        If no vector in the LLL reduced basis meets the given tolerance `epsilon`.
    """

    # set up the lattice basis
    signal = mp_real(mp_idft(dft))
    N = len(signal)
    if N == 1:
        return signal
    M = max(len(known_coeffs), 1)
    basis_matrix = _get_basis_matrix(signal, dft, known_coeffs=known_coeffs, inverted=inverted, factors=factors)
    
    # estimate what the tolerance should be
    precision = mp.get_context().precision
    if epsilon is None:
        epsilon = 10 ** max(-.3 * precision + 2.7 + .1 * N, -10)

    def check(vector):
        """Checks whether a given lattice vector is within the tolerance and has the correct B_0 block.
        """
        if len(vector) == N or vector[N] == beta0:
            return np.allclose(basis_matrix[N + 1:] @ np.concatenate([mp_round(vector[:N] + signal), [1]]), 0, atol=epsilon)

    # check the guess
    if check(np.zeros(N)):
        return mp_round(signal)

    # rescale the basis by parameters
    scaled_basis_matrix = np.copy(basis_matrix)
    scaled_basis_matrix[N] *= beta0
    scaled_basis_matrix[N + 1: -2 * M] *= beta1
    scaled_basis_matrix[-2 * M:] *= beta2
    scaled_basis_matrix *= beta3

    # find the LLL reduced basis and return the shortest vector within the tolerance
    reduced_basis = _approximate_svp(np.vectorize(int, otypes = [object])(scaled_basis_matrix).transpose(), delta=delta) / beta3
    for vector in reduced_basis:
        for sign in [-1, 1]:
            if check(sign * vector):
                return mp_round(sign * vector[:N] + signal)
    
    params = {'epsilon': epsilon, 'precision': precision, 'beta2': beta2}
    raise InversionError(f"Failure to recover a length {N} subproblem. It's possible that recovery was correct and the tolerance was too low. If you believer this is the case, try increasing epsilon. If recovery was incorrect, increasing precision and beta2 may aid in recovery.", **params)

class InversionError(Exception):

    """Python-exception-derived object raised by inversion functions.

    This exception is raised by ``invert_1D`` and ``invert_2D`` when the they fail to solve any subproblem to within the given tolerance. Contains current values of relevant parameters at the time of error.

    Parameters
    ----------
    msg: str
        The error message.
    beta2: float
        The value of the lattice parameter beta2.
    precision: int
        The number of bits of precision in the current `gmpy2 context`.
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
        
@my_vectorize(signature="(N)->(N)", excluded=set(range(1, len(_lll_params) + 1)) | {"known_coeffs"} | _lll_params)	
def invert_1D(signal, known_coeffs={}, **lattice_params):
    r"""Invert an integer signal from limited DFT spectrum.

    Invert the last axis of an integer signal from a limited set of sampled DFT coefficients. The sampled frequencies may be provided in `known_coeffs`, which should be structured like the output of `select_coeffs_1D`. The input `signal` should be given in real space, so the known DFT coefficients are obtained by calling `mp_dft` on `signal`. If no known frequencies are provided, they are chosen automatically from `signal` by assuming nonzero DFT coefficients are known. It is assumed that a sufficient set of coefficients are known to guarantee uniqueness of inversion. 

    Parameters
    ----------
    signal : mpfr arraylike
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

    See also
    --------
    sample_1D : constructing the sampled input
    select_coeffs_1D : selecting a partial set of known DFT coefficients
    invert_2D : analogous 2D function

    Other Parameters
    ----------------
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
    The parameters `beta0`, `beta1`, `beta2`, `beta3`, `delta`, and `epsilon` are passed as keyword arguments through `**lattice_params`. They control the lattice-based integer programming solver. 

    This dynamic programming implementation of 1D inversion iterates through the divisors :math:`d` of the signal size `N = len(signal)`. Each iteration requires solving an integer linear program in :math:`d` variables. The integer program is reduced to the shortest vector problem by constructing a lattice basis with reduction parameters :math:`\beta_0,\beta_1,\beta_2` [LV]_. This shortest vector problem is solved with the LLL approximation algorithm [LLL]_ using the given value of :math:`\delta`. The vector returned by LLL is rejected if the known part of its DFT does not match `signal` to absolute tolerance `epsilon`, causing an ``InversionError``.

    The default values are optimized for double precision. When using increased precision for larger inversions, it may be necessary to increase `beta2` and `beta3` and decrease `epsilon`.


    Examples
    --------

    Sampling and inverting with automatically selected coefficients:

    >>> signal = np.array([1, 1, 0, 1, 0, 1, 0])
    >>> sampled = intvert.sample_1D(signal)
    >>> np.allclose(signal, intvert.invert_1D(sampled))
    True

    Sampling and inverting with user-selected coefficients:

    >>> known_coeffs = {7: {0}, 1: {2}}
    >>> sampled = intvert.sample_1D(signal, known_coeffs)
    >>> np.allclose(signal, intvert.invert_1D(sampled, known_coeffs))
    True
    
    Sampling with user selection and inverting with automatically selected coefficients:

    >>> np.allclose(signal, intvert.invert_1D(sampled))
    True
    
    Inverting a larger example:

    >>> signal = np.random.randint(0, 2, 30)
    >>> signal # doctest: +SKIP
    array([1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 
        0, 1, 1, 0, 1, 0, 0, 0]) # random
    >>> sampled = intvert.sample_1D(signal)
    >>> np.allclose(signal, intvert.invert_1D(sampled))
    True

    With insufficient precision, inversion may fail for long signals with large integers:

    >>> signal = np.arange(29)
    >>> sampled = intvert.sample_1D(signal)
    >>> np.allclose(signal, intvert.invert_1D(sampled))
    False

    In this case, increasing precision and beta2 allows inversion:

    >>> with gmpy2.get_context() as c:
    ...     c.precision = 100
    ...     sampled = intvert.sample_1D(signal)
    ...     np.allclose(signal, intvert.invert_1D(sampled, beta2=1e20))
    ... 
    True

    Or providing more DFT coefficients also allows inversion:

    >>> known_coeffs = intvert.select_coeffs_1D(29, 2)
    >>> sampled = intvert.sample_1D(signal, known_coeffs)
    >>> np.allclose(signal, intvert.invert_1D(sampled, known_coeffs))
    True
    """

    N = len(signal)
    dft = mp_dft(signal)
    inverted = {1: mp_real(dft[:1])}
    for d in sp.divisors(N)[1:]:

        current_coeffs = [k * d // N for k in known_coeffs[N // d]] if d in known_coeffs else []
        factors = sp.primefactors(d)

        inverted[d] = _setup_and_solve(dft[:N:N // d], inverted=inverted, known_coeffs=current_coeffs, factors=factors, **lattice_params)

    return inverted[N].astype(int)
    
@my_vectorize(signature="(M,N)->(M,N)", excluded=set(range(1, len(_lll_params) + 1)) | {"known_coeffs"} | _lll_params)	
def invert_2D(signal, known_coeffs={}, **lattice_params):
    r"""Invert an integer matrix from limited DFT spectrum.

    Invert the last two axes of an integer signal from a limited set of sampled DFT coefficients. The sampled frequencies may be provided in `known_coeffs`, which should be structured like the output of `select_coeffs_2D`. The input `signal` should be given in real space, so the known DFT coefficients are obtained by calling `mp_dft2` on `signal`. If no known frequencies are provided, they are chosen automatically from `signal` by assuming nonzero DFT coefficients are known. It is assumed that a sufficient set of coefficients are known to guarantee uniqueness of inversion. 

    Parameters
    ----------
    signal : mpfr arraylike
        Sampled 2D signal.
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


    See also
    --------
    sample_2D : constructing the sampled input
    select_coeffs_2D : selecting a partial set of known DFT coefficients
    invert_1D : analogous 1D function

 
    Other Parameters
    ----------------
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
    Let :math:`(N_1, N_2)` be the shape of the last two axes of signal. This dynamic programming implementation of 2D inversion iterates through pairs of divisors of :math:`N_1, N_2`, with several 1D inversions occuring at each iteration. For details on 1D inversion and the keyword parameters `**lattice_params`, see ``invert_1D``.


    Examples
    --------

    Sampling and inverting with automatically selected coefficients:

    >>> signal = np.arange(10).reshape((2, 5))
    >>> sampled = intvert.sample_2D(signal)
    >>> np.allclose(intvert.invert_2D(sampled), signal)
    True

    Sampling and inverting with user-selected coefficients:

    >>> known_coeffs = {(2, 5): {frozenset({(0, 0)})}, (2, 1): {frozenset({(0, 4)})}, (1, 5): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 4)})}}
    >>> sampled = intvert.sample_2D(signal, known_coeffs)
    >>> np.allclose(intvert.invert_2D(sampled, known_coeffs), signal)
    True
    
    Sampling with user selection and inverting with automatically selected coefficients:

    >>> np.allclose(intvert.invert_2D(sampled), signal)
    True
    
    Inverting a larger example:

    >>> np.random.seed(0)
    >>> signal = np.random.randint(0, 2, (30, 20))
    >>> sampled = intvert.sample_2D(signal)
    >>> np.allclose(intvert.invert_2D(sampled), signal)
    True

    With insufficient precision, inversion may fail for long signals with large integers:

    >>> signal = np.random.randint(0, 2, (29, 29))
    >>> sampled = intvert.sample_2D(signal)
    >>> try:
    ...		np.allclose(intvert.invert_2D(sampled), signal)
    ... except intvert.InversionError as err:
    ...		err
    ...
    InversionError("Failure to recover a length 29 subproblem. It's possible that recovery was correct and the tolerance was too low. If you believer this is the case, try increasing epsilon. If recovery was incorrect, increasing precision and beta2 may aid in recovery.")

    In this case, increasing precision and beta2 allows inversion:

    >>> with gmpy2.get_context() as c:
    ...     c.precision = 100
    ...     sampled = intvert.sample_2D(signal)
    ...     np.allclose(intvert.invert_2D(sampled, beta2=1e16), signal)
    ... 
    True

    Or providing more DFT coefficients also allows inversion:

    >>> known_coeffs = intvert.select_coeffs_2D(29, 29, 2)
    >>> sampled = intvert.sample_2D(signal, known_coeffs)
    >>> np.allclose(intvert.invert_2D(sampled, known_coeffs), signal)
    True
    """

    M, N = signal.shape
    dft = mp_dft2(signal)
    dsums = {} 
    
    all_coeff_classes = get_coeff_classes_2D(M, N, False)

    for N1, N2 in product(sp.divisors(M), sp.divisors(N)):

        len = np.lcm(N1, N2)
        factors = sp.primefactors(len)
        lams = np.arange(len)

        for class_ in all_coeff_classes[M // N1, N // N2]:

            # find the set of known_coeffs in the current class
            for coeffs in known_coeffs[M // N1, N // N2] if (M // N1, N // N2) in known_coeffs else {}:
                if class_ & coeffs: break
            else:
                coeffs = []

            k, l = set(class_).pop()
            k1, l1 = k * lams % M, l * lams % N
            
            # find the immediate subproblems
            inverted = {}
            for p in factors:
                
                kp, lp = k1[p % len], l1[p % len]
                inverted[len // p] = dsums[kp, lp]

            direction_dft = dft[k1, l1]

            lll_result = _setup_and_solve(direction_dft, inverted=inverted, known_coeffs=[lam for lam in lams if (k1[lam], l1[lam]) in coeffs], factors=factors, **lattice_params)

            # update the DFT and memo
            dft[k1, l1] = mp_dft(lll_result)

            if len == np.lcm(M, N):
                continue

            permutations = {} 
            for lam in lams:
                if np.gcd(lam, len) == 1:
                    lam_inv = pow(int(lam), -1, int(len))
                    permutations[k1[lam], l1[lam]] = lll_result[lam_inv * np.arange(len) % len]

            dsums.update(permutations)

    return mp_round(mp_real(mp_idft2(dft))).astype(int)
