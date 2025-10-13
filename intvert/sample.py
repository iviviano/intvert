import numpy as np
import gmpy2 as mp
import sympy as sp
from itertools import product, chain
from functools import wraps

def my_vectorize(**vecargs):
    """Apply np.vectorize to a py func and keep its docstring.
    """
    def helper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return np.vectorize(func, **vecargs)(*args, **kwargs)

        return wrapper

    return helper

_cache = {}

@my_vectorize(signature="(n)->(n)")
def mp_dft(signal):
    r"""Compute the 1D discrete Fourier transform of a signal.

    This function computes the 1D discrete Fourier transform (DFT) along the last axis `signal` at the precision specified by the current `gmpy2 context`.

    Parameters
    ----------
    signal : array_like 
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 1D DFT of `signal` along the last axis.

    See also
    --------
    mp_idft : inverse of ``mp_dft``
    mp_dft2 : analogous 2D function

    Notes
    -----
    The algorithm used by this procedure depends on the current precision of the `gmpy2 context`. If the current precision is at most double, the input is cast to numpy floats and the Fast Fourier transform is computed with the :py:func:`numpy:numpy.fft.fft` procedure. If the precision of the current context is greater than 53 bits, the DFT is computed with standard matrix-vector multiplication.

    The DFT convention used in this implementation is

    .. math:: 
        \tilde{x}_k = \sum_{n = 0}^{N - 1}x_n e^{-2\pi{\rm i}nk/N},

    where :math:`{\bf x}` is an array of length `N`. The inverse DFT is given by

    .. math:: 
        x_n = \frac{1}{N}\sum_{k = 0}^{N - 1}\tilde{x}_k e^{2\pi{\rm i}nk/N}.

    """

    precision = mp.get_context().precision
    if precision <= 53: # with <= double precision, use np.fft
        return np.fft.fft(signal.astype(complex)).astype(mp.mpc)

    N = len(signal)

    # otherwise, matrix multiplication
    last_prec = -1
    if N in _cache:
        matrix, last_prec = _cache[N]
    if last_prec != precision:
        # precompute the N possible N'th roots of unity
        roots = [mp.root_of_unity(N, (N - 1) * n % N) for n in range(N)]
        matrix = np.array([[roots[n * k % N] for k in range(N)] for n in range(N)])
        _cache[N] = (matrix, precision)

    return matrix @ signal 

@my_vectorize(signature="(m,n)->(m,n)")
def mp_dft2(signal):
    r"""Compute the 2D discrete Fourier transform of a signal.

    This function computes the 2D discrete Fourier transform (DFT) along the last two axes of `signal` at the precision specified by the current `gmpy2 context`.

    Parameters
    ----------
    signal : array_like 
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 2D DFT of `signal` along the last two axes.

    See also
    --------
    mp_idft2 : inverse of ``mp_dft2``
    mp_dft : analogous 1D function

    Notes
    -----
    This function is implemented with ``mp_dft``. For algorithm notes, see ``mp_dft``.

    The DFT convention used in this implementation is

    .. math:: 
        \tilde{X}_{kl} = \sum_{m = 0}^{N_1 - 1}\sum_{n = 0}^{N_2 - 1}X_{mn} e^{-2\pi{\rm i}(mk/N_1 + nl/N_2)},

    where :math:`{\bf X}` is an :math:`N_1 \times N_2` matrix. The inverse DFT is given by

    .. math:: 
        X_{mn} = \sum_{k = 0}^{N_1 - 1}\sum_{l = 0}^{N_2 - 1}\tilde{X}_{kl} e^{2\pi{\rm i}(mk/N_1 + nl/N_2)}.

    """

    return mp_dft(mp_dft(signal).T).T

def mp_idft(signal):
    r"""Compute the 1D inverse discrete Fourier transform of a signal.

    This function computes the inverse of the 1D discrete Fourier transform along the last axis of `signal`, at the precision specified by the current `gmpy2 context`. 

    Parameters
    ----------
    signal : array_like
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 1D inverse DFT of `signal` along the last axis.

    See also
    --------
    mp_dft : inverse of ``mp_idft``
    mp_idft2 : analogous 2D function

    Notes
    -----
    This function is implemented with ``mp_dft``. For conventions and algorithm notes, see ``mp_dft``.

    Examples
    --------

    When more than 53 bits of precision are used, ``mp_dft`` and ``mp_idft`` yield higher numerical accuracy than :py:func:`numpy:numpy.fft.fft` and :py:func:`numpy:numpy.fft.ifft`:

    >>> signal = np.arange(19)
    >>> max(abs(np.fft.ifft(np.fft.fft(signal)) - signal))
    np.float64(3.552713678800501e-15)
    >>> with gmpy2.context() as c:
    ...     c.precision = 200
    ...     max(abs(intvert.mp_idft(intvert.mp_dft(signal)) - signal))
    ... 
    mpfr('2.994393295486490319393192740125243605675578124979580666244333e-59',200)

    """

    return np.conj(mp_dft(np.conj(signal))) / signal.shape[-1]

def mp_idft2(signal): 
    r"""Compute the 2D inverse discrete Fourier transform of a signal.

    This function computes the inverse of the 2D discrete Fourier transform along the last two axes of `signal`, at the precision specified by the current `gmpy2 context`. 

    Parameters
    ----------
    signal : array_like
        Input signal, can be complex.

    Returns
    -------
    mpc ndarray
        The 2D inverse DFT of `signal` along the last two axes.

    See also
    --------
    mp_dft2 : inverse of ``mp_idft2``
    mp_idft : analogous 1D function

    Notes
    -----
    This function is implemented with ``mp_dft2``. For conventions and algorithm notes, see ``mp_dft2``.

    Examples
    --------

    When more than 53 bits of precision are used, ``mp_dft2`` and ``mp_idft2`` yield higher numerical accuracy than :py:func:`numpy:numpy.fft.fft2` and :py:func:`numpy:numpy.fft.ifft2`:

    >>> signal = np.arange(30).reshape(5, 6)
    >>> np.max(abs(np.fft.ifft2(np.fft.fft2(signal)) - signal))
    np.float64(7.401486830834377e-17)
    >>> with gmpy2.get_context() as c:
    ...     c.precision = 200
    ...     np.max(abs(intvert.mp_idft2(intvert.mp_dft2(signal)) - signal))
    ... 
    mpfr('1.9957852381702224870373111645968013799923440139299733596839611e-59',200)
    """
    
    return np.conj(mp_dft2(np.conj(signal))) / np.prod(signal.shape[-2:])


mp_real = np.vectorize(lambda x: x.real, doc="Vectorized real part of an mpc np.ndarray")
mp_imag = np.vectorize(lambda x: x.imag, doc="Vectorized imaginary part of an mpc np.ndarray")
mp_round = np.vectorize(lambda x: mp.rint(x), doc="Vectorized round an mpfr np.ndarray to integers")


def _to_1D(coeff_classes_2D):
    """Restructure a dictionary of 2D DFT equivalence classes as 1D, assuming N = 1
    """

    return {divisor: {k for k, _ in coeff_classes_2D[divisor, 1].pop()} for   divisor, _ in coeff_classes_2D}

 
# @set_module("intvert")
def get_coeff_classes_1D(N, include_conjugates=True):
    r"""Returns a dictionary of classes of DFT coefficient frequencies for a 1D integer signal.

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

    See also
    --------
    select_coeffs_1D : selecting a partial set of DFT coefficients sufficient for inversion
    get_coeff_classes_2D : analogous 2D function

    Notes
    -----
    If :math:`{\bf x}` is an integer signal of length `N`, two DFT coefficients :math:`\tilde{x}_k` and :math:`\tilde{x}_l` are equivalent if :math:`\gcd(k, N)=\gcd(l, N)` [PC]_. Assuming :math:`{\bf x}` is real implies :math:`\tilde{x}_k = \tilde{x}_{N - k}^*`, so these DFT coefficients are trivially equivalent.

    Examples
    --------
    >>> intvert.get_coeff_classes_1D(6)
    {6: {0}, 1: {1, 5}, 2: {2, 4}, 3: {3}}

    >>> intvert.get_coeff_classes_1D(6, include_conjugates=False)
    {6: {0}, 1: {1}, 2: {2}, 3: {3}}
    """
    
    return _to_1D(get_coeff_classes_2D(N, 1, include_conjugates=include_conjugates))
    
 
# @set_module("intvert")
def get_coeff_classes_2D(M, N, include_conjugates=True):
    r"""Returns a dictionary of classes of DFT coefficient frequencies for a 2D integer matrix.

	Constructs a dictionary describing the equivalence classes of DFT coefficient frequencies for a 2D integer signal. The dictionary is structured as follows. It maps pairs :math:`(D_1, D_2)` of divisors of `M` and `N` to sets of coefficient classes. Each coefficient class is a `FrozenSet` of frequency pairs :math:`(k, l)`. A coefficient class containing :math:`(k, l)` is in the set mapped to by :math:`(D_1, D_2)` whenever :math:`\gcd(k, M) = D_1` and :math:`\gcd(l, N) = D_2`. If `include_conjugates` is `False`, frequencies made redundant by the signal being real are excluded. Otherwise, all frequences are included.

    Parameters
    ----------
    M : int
        Height of the matrix.
    N : int
        Width of the matrix.
    include_conjugates : bool, optional
        Whether to include coefficients made redundant by the signal being real, by default True

    Returns
    -------
    Dict[Tuple[int, int], Set[FrozenSet[Tuple[int, int]]]]
        Dictionary mapping pairs of divisors of `M` and `N` to sets of coefficient classes.

    See also
    --------
    select_coeffs_2D : selecting a partial set of DFT coefficients sufficient for inversion
    get_coeff_classes_1D : analogous 1D function

    Notes
    -----
    If :math:`{\bf X}` is an integer matrix of size `M, N`, two DFT coefficients :math:`\tilde{X}_{kl}` and :math:`\tilde{X}_{k'l'}` are equivalent if there exists `\lambda` relatively prime with `M` and `N` such that :math:`k = \lambda k' \pmod{M}` and :math:`l = \lambda l' \pmod{N}` [LV]_. Assuming :math:`{\bf X}` is real implies :math:`\tilde{X}_{kl} = \tilde{X}_{M - k, N - l}^*`, so these DFT coefficients are trivially equivalent.

    Examples
    --------
    >>> intvert.get_coeff_classes_2D(4, 4)
    {(4, 4): {frozenset({(0, 0)})}, (4, 1): {frozenset({(0, 1), (0, 3)})}, (4, 2): {frozenset({(0, 2)})}, (1, 4): {frozenset({(1, 0), (3, 0)})}, (1, 1): {frozenset({(3, 1), (1, 3)}), frozenset({(1, 1), (3, 3)})}, (1, 2): {frozenset({(3, 2), (1, 2)})}, (2, 4): {frozenset({(2, 0)})}, (2, 1): {frozenset({(2, 3), (2, 1)})}, (2, 2): {frozenset({(2, 2)})}}

    >>> intvert.get_coeff_classes_2D(4, 4, False)
    {(4, 4): {frozenset({(0, 0)})}, (4, 1): {frozenset({(0, 1)})}, (4, 2): {frozenset({(0, 2)})}, (1, 4): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 1)}), frozenset({(1, 3)})}, (1, 2): {frozenset({(1, 2)})}, (2, 4): {frozenset({(2, 0)})}, (2, 1): {frozenset({(2, 1)})}, (2, 2): {frozenset({(2, 2)})}}
    """
        
    found = np.zeros((M, N), dtype=bool)

    classes = {}
    for k, l in product(range(M), range(N)):

        if found[k, l]:
            continue

        gcd = int(np.gcd(k, M)), int(np.gcd(l, N))

        eclass = frozenset((k * lam % M, l * lam % N) for lam in range(np.lcm(M, N)) if np.gcd(lam, N * M) == 1)
        
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
    """Level of the subgroup generated by `(k, l)` in the lattice of cyclic subgroups of Z_M x Z_N.

    Levels are 1-indexed with level 1 containing the identity and <(1, 1)> on the highest level.
    """

    order_M = M // np.gcd(k, M)
    order_N = N // np.gcd(l, N)
    order = np.lcm(order_M, order_N)
    return sum(sp.factorint(order).values()) + 1


def select_coeffs_1D(N, Ls=[]):
    r"""Selects a set of DFT coefficient frequencies.

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

    See also
    --------
    get_coeff_classes_1D : structure of full DFT coefficient equivalence classes
    sample_1D : use the selected coefficients to sample an integer signal
    select_coeffs_2D : analogous 2D function

    Examples
    --------
    By default, get 1 element of each coefficient class

    >>> intvert.select_coeffs_1D(10)
    {10: {0}, 1: {1}, 2: {2}, 5: {5}} 

    If Ls is an integer, there are Ls generators in every class with with more than one generator between 0 and N / 2.

    >>> intvert.select_coeffs_1D(10, 2)
    {10: {0}, 1: {1, 3}, 2: {2, 4}, 5: {5}} 

    If Ls is a list of length 2, all classes on the top two levels of the lattice may have up to two generators. This is realized for d = 1 and d = 2. However, there is only one generator for the subgroup corresponding to d = 5.

    >>> intvert.select_coeffs_1D(10, [2]) 
    {10: {0}, 1: {1, 3}, 2: {2}, 5: {5}}
    """ 

    return _to_1D(select_coeffs_2D(N, 1, Ls))


def select_coeffs_2D(M, N, Ls = []):
    r"""Selects a set of DFT coefficient frequencies.

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
    selected: Dict[Tuple[int, int], Set[FrozenSet[Tuple[int, int]]]]
        Dictionary of selected coefficients.

    See also
    --------
    get_coeff_classes_2D : structure of full DFT coefficient equivalence classes
    sample_2D : use the selected coefficients to sample an integer matrix
    select_coeffs_1D : analogous 1D function

    Examples
    --------

    By default, get 1 element of each coefficient class:

    >>> intvert.select_coeffs_2D(2, 10)
    {(2, 10): {frozenset({(0, 0)})}, (2, 1): {frozenset({(0, 1)})}, (2, 2): {frozenset({(0, 2)})}, (2, 5): {frozenset({(0, 5)})}, (1, 10): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 1)})}, (1, 2): {frozenset({(1, 2)})}, (1, 5): {frozenset({(1, 5)})}}

    If Ls is an integer, there are up to Ls generators in every lattice class:

    >>> intvert.select_coeffs_2D(2, 10, 2)
    {(2, 10): {frozenset({(0, 0)})}, (2, 1): {frozenset({(0, 1), (0, 3)})}, (2, 2): {frozenset({(0, 2), (0, 4)})}, (2, 5): {frozenset({(0, 5)})}, (1, 10): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 1), (1, 3)})}, (1, 2): {frozenset({(1, 2), (1, 4)})}, (1, 5): {frozenset({(1, 5)})}}

    This is realized for the classes generated by `(0, 1)`, `(0, 2)`, `(1, 1)`, and `(1, 2)`. However, there is only one generator (accounting for real DFT symmetry) for the other classes.


    If Ls is a list of length 1, all classes on the top level of the lattice may have up to two generators:

    >>> intvert.select_coeffs_2D(2, 10, [2])
    {(2, 10): {frozenset({(0, 0)})}, (2, 1): {frozenset({(0, 1), (0, 3)})}, (2, 2): {frozenset({(0, 2)})}, (2, 5): {frozenset({(0, 5)})}, (1, 10): {frozenset({(1, 0)})}, (1, 1): {frozenset({(1, 1), (1, 3)})}, (1, 2): {frozenset({(1, 2), (1, 4)})}, (1, 5): {frozenset({(1, 5)})}}
    

    """ 

	# Set up list of number of coefficients by lattice depth
    lattice_depth = _get_lattice_level(1, 1, M, N)
    try:
        Ls = [L for L in Ls] + [1] * (lattice_depth - len(Ls))
    except TypeError:
        Ls = [Ls] * lattice_depth

	# Select coefficients from set of all classes
    all_selected_coeffs = {}
    for (d1, d2), classes in get_coeff_classes_2D(M, N, include_conjugates=False).items():
        all_selected_coeffs[d1, d2] = set()
        for coeff_class in classes:
            coeff_class = list(sorted(coeff_class))
            k, l = coeff_class[0]
            L = Ls[-_get_lattice_level(k, l, M, N)]
            selected_coeffs = coeff_class[:L]
            all_selected_coeffs[d1, d2].add(frozenset(selected_coeffs))

    return all_selected_coeffs


@my_vectorize(signature="(N)->(N)", excluded={1, "known_coeffs"})
def sample_1D(signal, known_coeffs=None):
    r"""Sample DFT coefficients of a 1D integer signal.

    Samples a subset of the DFT coefficients of a 1D integer signal in frequency space. DFT coefficients besides the sampled ones are set to 0, and the inverse DFT is returned. If `known_coeffs` is given, the DFT frequencies in `known_coeffs` are sampled. Otherwise, a minimial set of DFT coefficients is sampled to ensure uniqueness of recovery. To make the returned signal real, conjugate pairs of DFT coefficients are both sampled, as described in Notes.

    Parameters
    ----------
    signal : arraylike
        Signal to be sampled.
    known_coeffs : Dict, optional
        Dictionary of coefficients to sample, structured as in ``get_coeff_classes_1D``, by default None

    Returns
    -------
    out: mpfr ndarray
        Real space signal sampled along last axis.

    See also
    --------
    select_coeffs_1D : selecting a partial set of known DFT coefficients
    invert_1D : inverting the sampled signal to recover the original integer signal
    sample_2D : analogous 2D function

    Notes
    -----
    Let :math:`N` be the signal size of the last axis of `signal`. If the frequency :math:`k` is to be sampled, the frequency :math:`N - k` is also sampled. This ensures that the sampled signal in real space is real-valued. If we write :math:`S` as the set of known DFT frequencies, which satisfies :math:`k \in S \implies N - k \in S`, the returned signal :math:`\overline{\bf x}` is given entrywise by

    .. math:: 
        \overline{x}_n = \sum_{k \in S} \tilde{x}_k e^{-2\pi{\rm i}nk/N}.
    """

    # Generate sampling mask
    N = len(signal)
    mask = np.zeros(N, dtype=bool)
    known_coeffs = np.array(sum(map(list, known_coeffs.values()), []) if known_coeffs else sp.divisors(N), dtype=int) % N
    mask[known_coeffs] = 1
    mask[-known_coeffs] = 1

    # Apply mask in frequency space
    dft = mp_dft(signal)
    dft[~mask] = 0
    return mp_real(mp_idft(dft))


@my_vectorize(signature="(M,N)->(M,N)", excluded={1, "known_coeffs"})
def sample_2D(signal, known_coeffs={}):
    r"""Sample DFT coefficients of a 2D integer signal.

    Samples a subset of the DFT coefficients of a 2D integer signal in frequency space. DFT coefficients besides the sampled ones are set to 0, and the inverse DFT is returned. If `known_coeffs` is given, the DFT frequencies in `known_coeffs` are sampled. Otherwise, a minimial set of DFT coefficients is sampled to ensure uniqueness of recovery. To make the returned signal real, conjugate pairs of DFT coefficients are both sampled, as described in Notes.

    Parameters
    ----------
    signal : arraylike
        Signal to be sampled.
    known_coeffs : Dict, optional
        Dictionary of coefficients to sample, structured as in ``get_coeff_classes_2D``, by default None

    Returns
    -------
    out: mpfr ndarray
        Real space signal sampled along the last two axes.

    See also
    --------
    select_coeffs_2D : selecting a partial set of known DFT coefficients
    invert_2D : inverting the sampled matrix to recover the original integer matrix
    sample_1D : analogous 1D function

    Notes
    -----
    Let :math:`(N_1, N_2)` be the size of the last two axes of `signal`. If the frequency :math:`(k, l)` is to be sampled, the frequency :math:`(N_1 - k, N_2 - l)` is also sampled. This ensures that the sampled signal in real space is real-valued. If we write :math:`S` as the set of known DFT frequencies, which satisfies :math:`(k, l) \in S \implies (N_1 - k, N_2 - l) \in S`, the returned signal :math:`\overline{\bf X}` is given entrywise by

    .. math:: 
        \overline{X}_{mn} = \sum_{(k, l) \in S} \tilde{X}_{kl} e^{-2\pi{\rm i}(mk/N_1 + nl/N_2)}.
    """

	# Generate sampling mask
    M, N = signal.shape
    known_coeffs = known_coeffs if known_coeffs else select_coeffs_2D(M, N)
    mask = np.zeros((M, N), dtype=bool)
    for coeff_class in chain(*known_coeffs.values()):
        for k, l in coeff_class:
            mask[k, l] = True	
            mask[-k, -l] = True	

    # Apply mask in frequency space
    dft = mp_dft2(signal)
    dft[~mask] = 0
    return mp_real(mp_idft2(dft))
