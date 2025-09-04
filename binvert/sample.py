import numpy as np
import gmpy2 as mp
import sympy as sp
from itertools import product, chain
from functools import wraps
from decorator import set_module

def my_vectorize(**kwargs):
    def helper(func):
        @wraps(func)
        @np.vectorize(**kwargs)
        @set_module("binvert")
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return helper

# @np.vectorize(signature="(n)->(n)")
@my_vectorize(signature="(n)->(n)")
def mp_dft(signal):
    """Compute the 1D discrete Fourier transform of a signal.

    This function computes the 1D discrete Fourier transform (DFT) along the last axis `signal`.

    Parameters
    ----------
    signal : array_like 
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 1D DFT of `signal` along the last axis.

    Notes
    -----
    The algorithm used by this procedure depends on the current precision of the `gmpy2` context. If the current precision is at most double, the input is cast to numpy floats and the Fast Fourier transform is computed with the `numpy.fft.fft` procedure. If the precision of the current context is greater than 53 bits, the DFT is computed with standard matrix-vector multiplication.

    The DFT convention used in this implementation is
    .. math:: \tilde{x}_k = \sum_{n = 0}^{N - 1}x_n e^{-2\pi{\rm i}nk/N},
    where :math:`{\bf x}` is an array of length `N`. The inverse DFT is given by
    .. math:: x_n = \frac{1}{N}\sum_{k = 0}^{N - 1}\tilde{x}_k e^{2\pi{\rm i}nk/N}.
    """

    if mp.get_context().precision <= 53:
        # print("Using np.fft")
        return np.fft.fft(signal.astype(complex)).astype(mp.mpc)

    N = len(signal)

    # root = mp.root_of_unity(N, N - 1)
    # return np.vander(root ** np.arange(N), N, increasing = True) @ signal
    return np.array([[mp.root_of_unity(N, (N - 1) * n * k % N) for k in range(N)] for n in range(N)]) @ signal

@my_vectorize(signature="(m,n)->(m,n)")
def mp_dft2(signal):
    """Compute the 2D discrete Fourier transform of a signal.

    This function computes the 2D discrete Fourier transform (DFT) along the last two axes of `signal`.

    Parameters
    ----------
    signal : array_like 
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 2D DFT of `signal` along the last two axes.

    Notes
    -----
    This function is implemented with `mp_dft`. For algorithm notes, see `mp_dft`.

    The DFT convention used in this implementation is
    .. math:: \tilde{x}_{k, l} = \sum_{m = 0}^{N_1 - 1}\sum_{n = 0}^{N_2 - 1}X_{m,n} e^{-2\pi{\rm i}(mk/N_1 + nl/N_2)},
    where :math:`{\bf X}` is an :math:`N_1 \times N_2` matrix. The inverse DFT is given by
    .. math:: X_{m, n} = \sum_{k = 0}^{N_1 - 1}\sum_{l = 0}^{N_2 - 1}\tilde{X}_{k,l} e^{2\pi{\rm i}(mk/N_1 + nl/N_2)},
    """


    intermediate = [mp_dft(row) for row in signal]
    return np.transpose([mp_dft(row) for row in np.transpose(intermediate)])

@set_module("binvert")
def mp_idft(signal):
    """Compute the 1D inverse discrete Fourier transform of a signal.

    This function computes the inverse of the 1D discrete Fourier transform along the last axis of `signal`, at the precision specified by the current `gmpy2` context. 

    Parameters
    ----------
    signal : array_like
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 1D inverse DFT of `signal` along the last axis.

    Notes
    -----
    This function is implemented with `mp_dft`. For conventions and algorithm notes, see `mp_dft`.

    Examples
    --------

    When more than 53 bits of precision are used, `mp_dft` and `mp_idft` yield higher numerical accuracy than `np.fft.fft` and `np.fft.ifft`:

    >>> signal = np.ones(101)
    >>> max(abs(np.fft.ifft(np.fft.fft(signal)) - signal))
    np.float64(1.0174987266641335e-15)
    >>> with gmpy2.get_context() as c:
    ...     c.precision = 200
    ...     max(abs(binvert.mp_idft(binvert.mp_dft(signal)) - signal))
    mpfr('1.866904583358342512143219216134037272177075650616350139930335e-60',200)
    """

    return np.conj(mp_dft(np.conj(signal))) / signal.shape[-1]

@set_module("binvert")
def mp_idft2(signal): 
    
    return np.conj(mp_dft2(np.conj(signal))) / np.prod(signal.shape[-2:])

real_doc = """Return the real part of a complex argument.

Parameters
----------
val: mpc array_like
Input mpc array or scalar.

Returns
-------
mpc ndarray or scalar
The real component of the complex argument. The type of val for the output will be mpfr.
"""

mp_real = np.vectorize(lambda x: x.real, doc=real_doc)
mp_imag = np.vectorize(lambda x: x.imag)
mp_round = np.vectorize(lambda x: mp.rint(x))


def _to_1D(coeff_classes_2D):

    return {divisor: {k for k, _ in coeff_classes_2D[divisor, 1].pop()} for   divisor, _ in coeff_classes_2D}

 
@set_module("binvert")
def get_coeff_classes_1D(N, include_conjugates=True):
    """Returns a dictionary of classes of DFT coefficient frequencies for a 1D integer signal.

    Constructs a dictionary mapping divisors of `N` to sets of equivalent DFT coefficient frequencies for a 1D integer signal of length `N`. The divisor d is mapped to a set of DFT frequencies containing all integers between 0 and `N` - 1 whose greatest common divisor with `N` is d. If `include_conjugates` is `False`, frequencies greater than or equal to `N` / 2 are excluded.

    Parameters
    ----------
    N : int
        Length of the signal.
    include_conjugates : bool, optional
        Whether to include coefficients made redundant by the signal being real, by default True

    Returns
    -------
    Dict[int, Set[int]]
        Dictionary mapping divisors of `N` to sets of equivalent frequencies.

    Notes
    -----
    If :math:`{\bf x}` is an integer signal of length `N`, two DFT coefficients :math:`\tilde{x}_k` and :math:`\tilde{x}_l` if :math:`\gcd(k, N)=\gcd(l, N)`. Assuming :math:`{\bf x}` is real implies :math:`\tilde{x}_k = \tilde{x}_{N - k}^*`, so these DFT coefficients are related trivially.

    Examples
    --------
    >>> binvert.get_coeff_classes_1D(6)
    {6: {0}, 1: {1, 5}, 2: {2, 4}, 3: {3}}

    >>> binvert.get_coeff_classes_1D(6, include_conjugates=False)
    {6: {0}, 1: {1}, 2: {2}, 3: {3}}
    """
    
    return _to_1D(get_coeff_classes_2D(N, 1, include_conjugates=include_conjugates))
    
 
@set_module("binvert")
def get_coeff_classes_2D(M, N, include_conjugates=True):
    found = np.zeros((M, N), dtype=bool)

    classes = {}
    for k, l in product(range(M), range(N)):

        if found[k, l]:
            continue

        # gcd_m, gcd_n = int(np.gcd(k, M)), int(np.gcd(l, N))
        # gcd = gcd_m, gcd_n
        gcd = int(np.gcd(k, M)), int(np.gcd(l, N))

        eclass = frozenset((k * lam % M, l * lam % N) for lam in range(M * N) if np.gcd(lam, N * M) == 1)
        
        for k, l in eclass:
            found[k, l] = True
        if not include_conjugates:
            eclass = frozenset((k, l) for k, l in eclass if k in [0, M / 2] and l <= N / 2 or 0 < k < M / 2)

        if gcd not in classes:
            classes[gcd] = {eclass}
        else:
            classes[gcd].add(eclass)


    return classes


def _get_lattice_level(k, l, M, N=1): # levels indexed 1, 2, ... starting at top level with coefficient (0, 0)

    order_M = M // np.gcd(k, M)
    order_N = N // np.gcd(l, N)
    order = np.lcm(order_M, order_N)
    return sum(sp.factorint(order).values()) + 1


    # Constructs a dictionary mapping divisors of `N` to sets of equivalent DFT coefficient frequencies for an integer signal of length `N`. The divisor d is mapped to a set of DFT frequencies containing integers between 0 and `N` - 1 whose greatest common divisor with `N` is d. The number of frequencies in this set is determined by `Ls`. If `Ls` is an integer, the number of frequencies in `selected[1]` is `Ls`, and each other set of frequencies has one element. If `Ls` is a list, `Ls[i]` is the number of frequencies in `selected[d]` if `d` generates a cyclic subgroup at the `i`'th level of the subgroup lattice of :math:`\mathbb{Z}_N`. If `Ls[i]` is larger than the number of generators of `selected[d]` which are between `0` and `N / 2`, `selected[d]` is just this maximal set of such generators.

@set_module("binvert")
def select_coeffs_1D(N, Ls=[]):
    """Selects a set of DFT coefficient frequencies.

    Constructs a dictionary mapping divisors of `N` to sets of equivalent DFT coefficient frequencies for an integer signal of length `N`. The divisor d is mapped to a set of DFT frequencies containing integers between 0 and `N` - 1 whose greatest common divisor with `N` is d. The number of frequencies in this set is determined by `Ls`. If `Ls` is an integer, the number of frequencies in set of frequencies will be at most `Ls`. If `Ls` is a list, `Ls[i]` is the number of frequencies in `selected[d]` if `d` generates a cyclic subgroup at the `i`'th level of the subgroup lattice of :math:`\mathbb{Z}_N`. If `Ls[i]` is larger than the number of generators of `selected[d]` which are between `0` and `N / 2`, `selected[d]` is just this maximal set of such generators.

    Parameters
    ----------
    N : int
        Length of the integer signal.
    Ls : int or list, optional
        Number of coefficients to include for each class, by default []

    Returns
    -------
    selected: Dict[int, Set[int]]
        Dictionary of selected coefficients.

    Examples
    --------
    >>> binvert.select_coeffs_1D(10)
    {10: {0}, 1: {1}, 2: {2}, 5: {5}} 

    By default, get 1 element of each coefficient class

    >>> binvert.select_coeffs_1D(10, 2)
    {10: {0}, 1: {1, 3}, 2: {2, 4}, 5: {5}} 

    If Ls is an integer, there are Ls generators in every class with with more than one generator between 0 and N / 2.

    >>> binvert.select_coeffs_1D(10, [2]) 
    {10: {0}, 1: {1, 3}, 2: {2, 4}, 5: {5}}
    
    If Ls is a list of length 2, all classes on the top two levels of the lattice may have up to two generators. This is realized for d = 1 and d = 2. However, there is only one generator for the subgroup corresponding to d = 5.
    """ 

    return _to_1D(select_coeffs_2D(N, 1, Ls))


@set_module("binvert")
def select_coeffs_2D(M, N, Ls = []):
    """Selects a set of DFT coefficient frequencies.

    Constructs a dictionary mapping divisors of `M` and `N` to sets of equivalent DFT coefficient frequencies for an `(M,N)` integer matrix. The pair of divisors `(d1, d2)` is mapped to a set of frozensets of DFT frequencies. Each frozenset contains equivalent frequencies pairs `(k, l)` where the greatest common divisor of `k` with `M` is `d1` and the greatest common divisor of `l` with `N` is `d2`. The number of frequencies in each frozen set is determined by `Ls`. If `Ls` is an integer, the number of frequencies in set of frequencies will be at most `Ls`. If `Ls` is a list, `Ls[i]` is the maximum number of frequencies in `selected[d1, d2]` if `(d1, d2)` generates a cyclic subgroup at the `i`'th level of the cyclic subgroup lattice of :math:`\mathbb{Z}_M\times\mathbb{Z}_N`. 

    Parameters
    ----------
    M : int
        First dimension of the integer signal.
    N : int
        Second dimension of the integer signal.
    Ls : int or list, optional
        Number of coefficients to include for each class, by default []

    Returns
    -------
    selected: Dict[int, Set[Frozenset[Tuple[int, int]]]]
        Dictionary of selected coefficients.

    Examples
    --------
    >>> binvert.select_coeffs_2D(2, 10)
    {(2, 10): {frozenset({(0, 0)})}, (1, 1): {frozenset({(1, 1)})}, (2, 1): {frozenset({(0, 1)})}, (1, 2): {frozenset({(1, 2)})}, (1, 5): {frozenset({(1, 5)})}, (1, 10): {frozenset({(1, 0)})}}}, (2, 2): {frozenset({(0, 2)})}, (2, 5): {frozenset({(0, 5)})}} 

    By default, get 1 element of each coefficient class

    >>> binvert.select_coeffs_2D(2, 10, 2)
    {(2, 10): {frozenset({(0, 0)})}, (2, 1): {frozenset({(0, 1), (0, 3)})}, (2, 2): {frozenset({(0, 2), (0, 4)})}, (2, 5): {frozenset({(0, 5)})}, (1, 10): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 1), (1, 3)})}, (1, 2): {frozenset({(1, 2), (1, 4)})}, (1, 5): {frozenset({(1, 5)})}}

    If Ls is an integer, there are Ls generators in the top level of the subgroup lattice class

    >>> binvert.select_coeffs_2D(2, 10, [2, 2]) 
    {(2, 10): {frozenset({(0, 0)})}, (2, 1): {frozenset({(0, 1), (0, 3)})}, (2, 2): {frozenset({(0, 2), (0, 4)})}, (2, 5): {frozenset({(0, 5)})}, (1, 10): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 1), (1, 3)})}, (1, 2): {frozenset({(1, 2), (1, 4)})}, (1, 5): {frozenset({(1, 5)})}}
    
    If Ls is a list of length 2, all classes on the top two levels of the lattice may have up to two generators. This is realized for the classes generated by `(1, 1)`, `(1, 2)`, `(0, 1)`, and `(0, 2)`. However, there is only one generator for the classes generated by `(0, 5)`, `(1, 5)`, and `(1, 0)`.
    """ 

    lattice_depth = _get_lattice_level(1, 1, M, N)
    assert lattice_depth == sum(sp.factorint(np.lcm(M, N)).values()) + 1

    try:
        # Ls = [L for L in Ls]
        Ls = [L for L in Ls] + [1] * (lattice_depth - len(Ls))
    except TypeError:
        # Ls = [Ls]
        Ls = [Ls] * lattice_depth
    # finally:
    #     Ls = Ls + [1] * (lattice_depth - len(Ls))

    all_selected_coeffs = {}

    for (d1, d2), classes in get_coeff_classes_2D(M, N, include_conjugates=False).items():
        all_selected_coeffs[d1, d2] = set()
        for coeff_class in classes:
            coeff_class = list(sorted(coeff_class))
            k, l = coeff_class[0]
            # print(f"k = {k}, l = {l}; level = {get_lattice_level(k, l, M, N)}")
            L = Ls[-_get_lattice_level(k, l, M, N)]
            # print(f"\t L = {L}")
            selected_coeffs = coeff_class[:L]
            all_selected_coeffs[d1, d2].add(frozenset(selected_coeffs))

    return all_selected_coeffs

@my_vectorize(signature="(N)->(N)", excluded={1, "known_coeffs"})
def sample_1D(signal, known_coeffs=None):
    """Sample DFT coefficients of a 1D integer signal.

    Samples a subset of the DFT coefficients of a 1D integer signal in frequency space. DFT coefficients besides the sampled ones are set to 0, and the inverse DFT is returned. If `known_coeffs` is given, the DFT frequencies in `known_coeffs` are sampled. Otherwise, a minimial set of DFT coefficients is sampled to ensure uniqueness of recovery. To make the returned signal real, conjugate pairs of DFT coefficients are both sampled, as described in Notes.

    Parameters
    ----------
    signal : ndarray
        Signal to be samples.
    known_coeffs : Dict, optional
        Dictionary of coefficients to sample, structured as in `get_coeff_classes_1D`, by default None

    Returns
    -------
    out: mpc ndarray
        Sampled signal in real space.

    Notes
    -----
    If the frequency :math:`k` is to be sampled, the frequency :math:`N - k` is also sampled. This ensures that the sampled signal is in real space is real-valued. If we write :math:`S` as the set of known DFT frequencies, which satisfies :math:`k \in S \implies N - k \in S`, the returned signal is given by
    .. math:: \overline{x}_n = \sum_{k \in S} \tilde{x}_k e^{-2\pi{\rm i}kn/N}.
    """

    N = len(signal)
    mask = np.zeros(N, dtype=bool)
    known_coeffs = np.array(sum(map(list, known_coeffs.values()), []) if known_coeffs else sp.divisors(N), dtype=int) % N
    mask[known_coeffs] = 1
    mask[-known_coeffs] = 1

    dft = mp_dft(signal)
    dft[~mask] = 0
    return mp_real(mp_idft(dft))

@my_vectorize(signature="(M,N)->(M,N)", excluded={1, "known_coeffs"})
def sample_2D(signal, known_coeffs={}):

    M, N = signal.shape

    known_coeffs = known_coeffs if known_coeffs else select_coeffs_2D(M, N)

    # print(known_coeffs.values())

    mask = np.zeros((M, N), dtype=bool)
    for coeff_class in chain(*known_coeffs.values()):
        # print(coeff_class)
        for k, l in coeff_class:
            mask[k, l] = True	
            mask[-k, -l] = True	

    dft = mp_dft2(signal)
    dft[~mask] = 0
    return mp_real(mp_idft2(dft))
