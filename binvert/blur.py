import numpy as np
import gmpy2 as mp
import sympy as sp
from itertools import product, chain

@np.vectorize(signature="(n)->(n)")
def mp_dft(signal):

    if mp.get_context().precision <= 53:
        # print("Using np.fft")
        return np.fft.fft(signal.astype(complex)).astype(mp.mpc)

    N = len(signal)
    root = mp.root_of_unity(N, N - 1)

    return np.vander(root ** np.arange(N), N, increasing = True) @ signal

@np.vectorize(signature="(m,n)->(m,n)")
def mp_dft2(signal):

    intermediate = [mp_dft(row) for row in signal]
    return np.transpose([mp_dft(row) for row in np.transpose(intermediate)])

def mp_idft(signal):

    return np.conj(mp_dft(np.conj(signal))) / signal.shape[-1]

def mp_idft2(signal): 
    
    return np.conj(mp_dft2(np.conj(signal))) / np.prod(signal.shape[-2:])

mp_real = np.vectorize(lambda x: x.real)
mp_imag = np.vectorize(lambda x: x.imag)
mp_round = np.vectorize(lambda x: mp.rint(x))

def _to_1D(coeff_classes_2D):

    return {divisor: {k for k, _ in coeff_classes_2D[divisor, 1].pop()} for   divisor, _ in coeff_classes_2D}

 
def get_coeff_classes_1D(N, include_conjugates=True):
    
    return _to_1D(get_coeff_classes_2D(N, 1, include_conjugates=include_conjugates))
    
 
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

def select_coeffs_1D(N, Ls=[]):

    return _to_1D(select_coeffs_2D(N, 1, Ls))


def select_coeffs_2D(M, N, Ls = []):

    lattice_depth = _get_lattice_level(1, 1, M, N)
    assert lattice_depth == sum(sp.factorint(np.lcm(M, N)).values()) + 1

    try:
        Ls = [L for L in Ls]
    except TypeError:
        Ls = [Ls]
    finally:
        Ls = Ls + [1] * (lattice_depth - len(Ls))

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

@np.vectorize(signature="(N)->(N)", excluded={1, "known_coeffs"})
def blur_1D(signal, known_coeffs=None):

    N = len(signal)
    mask = np.zeros(N, dtype=bool)
    known_coeffs = np.array(sum(map(list, known_coeffs.values()), []) if known_coeffs else sp.divisors(N), dtype=int) % N
    mask[known_coeffs] = 1
    mask[-known_coeffs] = 1

    dft = mp_dft(signal)
    dft[~mask] = 0
    return mp_real(mp_idft(dft))

@np.vectorize(signature="(M,N)->(M,N)", excluded={1, "known_coeffs"})
def blur_2D(signal, known_coeffs={}):

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
