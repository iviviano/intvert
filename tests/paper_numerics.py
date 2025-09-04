import numpy as np
import scipy
import gmpy2 as mp
import sympy as sp
import sys
import pandas as pd
import matplotlib.pyplot as plt
import unittest
import skimage
from PIL import Image
from itertools import product
from time import time

from context import binvert

import tracemalloc

def theoretical_beta2(N, M, upper_int, beta0=.1, K=None,):

    def vol(n):

        return np.pi ** (n / 2) / sp.functions.special.gamma_functions.gamma(n / 2 + 1)

    N_reduced = reduced(N)

    if K is None:
        K = .5 * ((N_reduced - 2 * M) * upper_int) ** .5

    first_term = vol(N_reduced) * (K ** 2 + beta0 ** 2) ** ((N_reduced + M) / 2) * ((N_reduced + 1) * vol(N_reduced) / (2 * N_reduced * vol(N_reduced - 1)))
    second_term = vol(N_reduced) * K ** (N_reduced + M) * sum(1 / np.arange(1, max(1, np.floor(((K / beta0) ** 2 + 1) ** .5))) ** M)
    return float(first_term + second_term) ** (1 / (2 * M))

def gmean(x, y):
    return (x * y) ** .5

def exp_search(step, target, key, n_max):
    current = step
    best_lower_bound = 1
    best_upper_bound = 0
    for n in range(n_max):
        value = key(current)
        # if value == target:
        #     return current, value
        # elif value < target:
        if value < target:
            best_lower_bound = current
            if best_upper_bound:
                current = gmean(best_upper_bound, current)
            else:
                current = step * current
        else:
            best_upper_bound = current
            current = gmean(best_lower_bound, current)
    return current, key(current)

def save_fig(fname):
    plt.savefig(f"../figures/{fname}.png", bbox_inches='tight')
    
def shallow_select_coeffs(N, M = 1):

    prime_factors = sp.factorint(N)
    if N in prime_factors:
        Ls = [M]
    else:
        Ls = [M] + [N] * sum(prime_factors.values())
    
    return binvert.select_coeffs_1D(N, Ls)

def reduced(N):
    return sp.totient(N)
    prime_factors = sp.primefactors(N)
    return int(np.round(N * np.prod(1 - 1 / np.array(prime_factors))))

class LatticeTheory(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(5748891)

    def test_K(self, N=30):

        upper_int = N

        Ms = np.arange(1, 4)
        # fig, ax = plt.subplots(1, len(Ms), figsize=(12, 5))
        fig, ax = plt.subplots(1, 1+ len(Ms), figsize=(12, 5))
        low = np.inf
        high = 0
        signals = self.rand.binomial(upper_int, .5, (int(1e5), N))
        for M in Ms:

            selected_coeffs = shallow_select_coeffs(N, M)
            blurred_signals = binvert.sample_1D(signals, selected_coeffs)
            Ks = np.linalg.norm(blurred_signals.astype(float) - signals, axis=1)

            kde = scipy.stats.gaussian_kde(Ks)
            x = np.linspace(min(Ks), max(Ks), 1000)

            hist = kde(x)

            ax[M - 1].plot(x, hist)
            ax[-1].plot(x, hist, label=f"$M = {M}$")

            theory = .5 * np.sqrt((int(reduced(N)) - 2 * M) * upper_int)
            ax[M - 1].vlines(theory, 0, kde(theory), color='red', label="Theoretical $K$", linestyle='dashed')
            mc = np.mean(Ks)
            ax[M - 1].vlines(mc, 0, kde(mc), color='blue', label="Average", linestyle='dashed')

            low = min(low, min(Ks))
            high = max(high, max(Ks))

        for ax, title in zip(ax.flatten(), ["$M = 1$", "$M = 2$", "$M = 3$", "All $M$"]):
            ax.set_xlabel(r"$K$")
            ax.set_xlim(low, high)
            ax.set_ylim(0, .25)
            ax.set_title(title)
        ax.legend()

        plt.subplots_adjust(wspace=0.6)

        save_fig(f"K_hist_N={N}")
        plt.show()

    def test_M_N(self):


        fig, ax = plt.subplots(2, 2)
        fig.set_size_inches(12, 9)

        for i, L in enumerate([1, None]):

            Ns = np.arange(19, 61)
            Ms = np.arange(1, 4)

            ca = ax[i, 1]

            for M in Ms:
                beta2s = []
                for N in Ns:
                    L = L if L else N
                    beta2s.append(theoretical_beta2(N, M, L))
                ca.plot(Ns, beta2s, label=f"$M = {M}$")
                ca.set_ylabel(r"$\beta_2$")
                ca.set_xlabel(r"$N$")
                ca.set_yscale('log')

                ca.legend()

            Ns = np.arange(29, 35)
            ca = ax[i, 0]
            for i, N in enumerate(Ns):
                L = L if L else N
                betas = []
                Ms = np.arange(1, (reduced(N)) // 2)
                for M in Ms:
                    betas.append(theoretical_beta2(N, M, L))
                ca.plot(Ms, betas, label=f"$N = {N}$")

                ca.set_xlabel(f"$M$")
                ca.set_ylabel(r"$\beta_2$")
                ca.set_yscale('log')

                ca.legend()

        plt.subplots_adjust(hspace=0.2, wspace=0.2)
        save_fig("M_N_dep")
        plt.show()

    def test_phi(self):

        N = np.arange(19, 61)

        phi = np.vectorize(sp.totient)(N)

        plt.xlabel(r"$N$")
        plt.ylabel(r"$\phi(N)$")

        plt.plot(N, phi)
        save_fig("phi")
        plt.show()

    
    def test_beta0(self):

        N = 30
        Ms = np.arange(1, 3)

        fig, ax = plt.subplots(1, len(Ms))
        fig.set_size_inches(12, 5)

        beta0s = np.concatenate([np.geomspace(1e-5, 1e-1), np.linspace(1e-1, 10)])
        for M in Ms:

            beta2s = []
            for beta0 in beta0s:
                beta2s.append(theoretical_beta2(N, M, N, beta0))

            ca = ax[M - 1]
            ca.plot(beta0s, beta2s)

            ca.set_title(f"$M = {M}$")
            ca.set_xlabel(r"$\beta_0$")
            ca.set_ylabel(r"$\beta_2$")
            ca.set_yscale('log')

        save_fig("beta0_dep")
        plt.show()

class Lattice(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(98708743)

        self.n_sig = 100
        self.Ns = [19, 30, 31, 32, 45, 47, 60]
        self.upper_int = None

        
    def test_beta_hist(self, n_betas = 10, N = 40, M=None):

        upper_int = self.upper_int if self.upper_int else N
        fname = f"../data/beta_hist_nsig={self.n_sig}_nbeta={n_betas}_L={self.upper_int}"
        results = pd.read_pickle(fname)

        with mp.get_context() as context:

            # context.precision = 400

            M = M if M else range(1, int(np.sqrt(N)))

            with self.subTest(N=N):

                signals = self.rand.binomial(upper_int, .5, (self.n_sig, N))

                for M in M:

                    selected_coeffs = shallow_select_coeffs(N, M)
                    print(selected_coeffs)
                    blurred = binvert.sample_1D(signals, selected_coeffs)

                    with self.subTest(M=M):

                        mag_guess = int(np.log10(float(theoretical_beta2(N, M, upper_int))))
                            
                        beta2s = np.logspace(mag_guess - 3, mag_guess + 3, n_betas)
                        
                        n_correct = []
                        for n_beta in range(n_betas):
                            n_correct.append(0)

                            beta2 = beta2s[n_beta]

                            for n in range(self.n_sig):
                                try:
                                    inverted = binvert.invert_1D(blurred[n], selected_coeffs, beta2=beta2)
                                    n_correct[-1] += np.allclose(signals[n] - inverted, 0)
                                except:
                                    pass

                            results[(N, M)].update({
                                "beta": beta2s,
                                "correct": n_correct,
                            })
                            results.to_pickle(fname)

                        print(f"N = {N}, M = {M}, correct: {n_correct}")

    
    def test_make_beta_hist(self, n_betas=10, N=40):

        upper_int = self.upper_int if self.upper_int else N

        fname = f"../data/beta_hist_nsig={self.n_sig}_nbeta={n_betas}_L={self.upper_int}"

        try:
            open(fname)

        except FileNotFoundError:

            results = {(N, M): {"beta": [0] * n_betas, "correct": [0] * n_betas} for N in range(1, 101) for M in range(1, int(np.sqrt(N)) + 1)}

            results = pd.DataFrame(results)

            results.to_pickle(fname)

        finally:

            results = pd.read_pickle(fname)

        print(results[N])

        Ms = range(1, int(np.sqrt(N)))

        fig, ax = plt.subplots(1, len(Ms), figsize=(12, 5))

        for M in Ms: 
            data = results[(N, M)]

            ax[M - 1].plot(data['beta'], np.array(data['correct']) / (self.n_sig * (N + 1)))

            ax[M - 1].set_xscale("log")
            ax[M - 1].set_xlabel(r"$\beta_2$")
            
            ax[M - 1].set_title(r"$M = %d$" % M)

            theory_beta2 = theoretical_beta2(N, M, upper_int)
            ax[M - 1].axvline(theory_beta2, color='red', linestyle='dashed', label="Theoretical $\\beta_2$")
        
        plt.suptitle(r"$N = %d$" % N)
        plt.subplots_adjust(wspace=0.6)
        save_fig("beta_hist_prime_n=%d" % self.n_sig)
        plt.show()

    def test_percentiles(self, step = 1e5, n_steps = 30, N = 19, targets = None, Ms=[], upper_int=None):

        fname = f"../data/percentiles_n={self.n_sig}_L={self.upper_int}"
        upper_int = upper_int if upper_int else self.upper_int if self.upper_int else N

        with mp.get_context() as context:

            results = pd.read_pickle(fname)

            context.precision = 200

            signals = self.rand.binomial(upper_int, .5, (self.n_sig, N))

            for M in Ms if Ms else range(1, 2):

                known_coeffs = shallow_select_coeffs(N, M)
                blurred_signals = binvert.sample_1D(signals, known_coeffs)

                with self.subTest(N=N, M=M):

                    def key(beta2):
                        correct = 0
                        for n in range(self.n_sig):
                            try:
                                inverted = binvert.invert_1D(blurred_signals[n], known_coeffs, beta2=beta2, epsilon=1e-11)
                                correct += np.allclose(inverted, signals[n])
                            except Exception:
                                pass
                            
                        return correct / self.n_sig

                    percentiles = []
                    targets = targets if targets else [50, 90, 100]
                    for target in targets:

                        percentile, value = exp_search(step, target / 100, key, n_steps)
                        percentiles.append(percentile)

                        print(f"{N}x{N}, M={M}, {target}th percentile: {percentile:.2e} (value: {value:.2f})")

                        
                        stderr = sys.stderr
                        with open("/dev/null", "w") as f:
                            sys.stderr = f
                            results[(N, M)].update({target: percentile})
                            results.to_pickle(fname)
                        sys.stderr = stderr

                print(results)

    def make_percentile_table(self):

        fname = f"../data/percentiles_n={self.n_sig}_L={self.upper_int}"

        try:

            open(fname)

        except FileNotFoundError:

            table = {(N, M): {
                "Theory": theoretical_beta2(N, M, self.upper_int if self.upper_int else N),
                50: 0,
                90: 0,
                100: 0
                } for N in range(19, 101) for M in range(1, 4)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname)

        print("\\begin{tabular}{cc|cccc}")

        header = "$N$ & $M$ & Theoretical $\\beta_2$"
        targets = [50, 90, 100]
        for target in targets:
            header += f" & {target}'th percentile $\\beta_2$"
        print(header + " \\\\")

        print("\midrule")

        # Ns = self.Ns
        # Ns = [19, 24, 25, 27, 30, 31, 32, 39, 45, 47, 49, 50, 60, 90]
        Ns = [19, 38] 

        pipe = '\\textbar'
        for N in Ns:

            print("\midrule")

            for M in range(1, 4):

                row = (pipe if M - 1 else f"${N}$") + f" & ${M}$"
                row += f" & {float(table[(N, M)].loc['Theory']):.02e}"

                for target in targets:
                    row += f" & {table[(N, M)].loc[target]:.2e}"

                print(row + " \\\\")

        print("\\end{tabular}")
    
    # def make_percentile_plot(self):

    #     table = pd.read_pickle(f"../data/percentiles_prime_n={self.n_sig}")

    #     # ax = table.plot(x="M", y=["Theory", 50, 90, 100], logy=True, subplots=True, layout=(1, len(self.Ns)), marker='o', title="Percentiles of $\\beta_2$ for prime $N$")
    #     fig, ax = plt.subplots(1, len(self.Ns), figsize=(12, 5))

    #     for i, N in enumerate(self.Ns):
    #         ax[i].set_title(r"$N = %d$" % N)
    #         print(table[N])
    #         ax[i].plot(table[N].T, label=["Theory", "50th", "90th", "100th"], marker='o')

    #         ax[i].set_yscale("log")
    #         ax[i].set_xlabel(r"$M$")
    #         ax[i].set_ylabel(r"$\beta_2$")
    #         ax[i].legend()

    #         ax[i].set_xticks(range(1, int(np.sqrt(N)) + 1))

    #     plt.subplots_adjust(wspace=0.6)

    #     save_fig("percentiles_n=%d" % self.n_sig)
    #     plt.show()

    def test_subproblem(self):

        self.upper_int = "subproblem"

        fname = f"../data/percentiles_n={self.n_sig}_L=subproblem"
        results = pd.read_pickle(fname)
        
        for N in [19, 38]:
            L = results[N, 1].loc['L']
            print(f"Testing N = {N}, L = {L}")
            self.test_percentiles(N=N, Ms=[1], upper_int=L)

    def make_subproblem_table(self):

        fname = f"../data/percentiles_n={self.n_sig}_L=subproblem"

        Ns = [19, 38]
        Ls = [40, 20]
        targets = [50, 90, 100]

        try:

            open(fname)

        except FileNotFoundError:

            table = {(N, M): {"L": L, 'theory': theoretical_beta2(N, M, L)} | {target: np.nan for target in targets} for N, L in zip(Ns, Ls) for M in range(1, 4)}

            table = pd.DataFrame(table)
            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname)

        print(table)
        print()

        print(r"\begin{tabular}{ccc|cccc}")

        header = "$N$ & $M$ & $L$ & Theoretical $\\beta_2$"
        for target in targets:
            header += f" & {target}'th percentile $\\beta_2$"

        print(header + "\\\\")

        print("\midrule")
        print("\midrule")

        for N, L in zip(Ns, Ls):

            for M in range(1, 2):

                print(f"{N} & {M} & {L} & {table[(N, M)].loc['theory']:.2e}", end='')

                for target in targets:
                    print(f" & {table[(N, M)].loc[target]:.2e}", end='')

                print("\\\\")

            print(r"\midrule")

        print(r"\end{tabular}")






    def test_K(self, N=range(92, 101)):

        fname = f"../data/K_n={self.n_sig}_L={self.upper_int}"
        
        try: 
            Ns = [N for N in N]
        except:
            Ns = [N]
        
        for N in Ns:
            upper_int = self.upper_int if self.upper_int else N

            with mp.get_context() as context:

                result = pd.read_pickle(fname)

                context.precision = 100

                L = int(np.sqrt(N))

                signals = self.rand.integers(0, 2, (self.n_sig, N))

                for  M in range(1, L + 1):
                    # blurred = blur.blur(matrices, l)

                    selected_coeffs = shallow_select_coeffs(N, M)
                    blurred = binvert.sample_1D(signals, selected_coeffs)

                    n_correct = 0
                    for n in range(self.n_sig):

                        try: 
                            beta2 = theoretical_beta2(N, M, upper_int)
                            inverted = binvert.invert_1D(blurred[n], selected_coeffs, beta2=beta2)
                            n_correct += np.allclose(inverted - signals[n], 0)
                        except:
                            pass

                    print(f"N={N}, M={M}, correct: {n_correct}")

                    result.loc[result['N'] == N, M] = n_correct
                    result.to_pickle(fname)


    def make_K_table(self):

        fname = f"../data/K_n={self.n_sig}_L={self.upper_int}"

        try:

            open(fname)

        except FileNotFoundError:
            Ns = range(1, 101)
            table = {"N": Ns} | {M: [np.nan] * len(Ns) for M in range(1, int(np.sqrt(max(self.Ns))) + 1)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname)

        print(table)

        print("\\begin{tabular}{c|%s}" % ("r" * (len(table.columns) - 1)))

        header = "\diagbox{$N$}{$M$}"
        # for M in range(1, int(np.sqrt(max(self.Ns))) + 1):
        for M in range(1, 4):
            header += f" & {M}"
        print(header + " \\\\")

        print("\\midrule")

        # pipe = '\\textbar'

        # for N in self.Ns:
        for N in range(1, 100):
            row = f"${N}$"
            # for M in range(1, int(np.sqrt(max(self.Ns))) + 1):
            for M in range(1, 4):
                value = table.loc[table['N'] == N, M].values[0]
                row += f" & {'|' if np.isnan(value) else value / (self.n_sig) * 100:.4}"
            print(row + " \\\\")

        print("\\end{tabular}")

    def test_prec(self, Ns=range(1, 101)):

        fname = f"../data/prec_n={self.n_sig}_L={self.upper_int}"
        result = pd.read_pickle(fname)

        beta2 = {23: 1e5, 53: 1e13}
        
        for N in Ns:
            upper_int = self.upper_int if self.upper_int else N
            signals = self.rand.binomial(upper_int, .5, (self.n_sig, N))

            for prec in [23, 53]:

                for M in range(1, 11):

                    with mp.get_context() as c:
                        c.precision = prec

                        known_coeffs = shallow_select_coeffs(N, M)
                        blurred = binvert.sample_1D(signals, known_coeffs)

                        correct = 0
                        for n in range(self.n_sig):

                            try:
                                inverted = binvert.invert_1D(blurred[n], known_coeffs, beta2=beta2[prec])
                                correct += np.allclose(signals[n] - inverted, 0)
                            except Exception:
                                pass

                        stderr = sys.stderr
                        with open("/dev/null", "w") as f:
                            sys.stderr = f
                            print(f"N = {N}; M = {M}; n_correct = {correct} (precision = {prec})")
                            result[prec][N, M] = correct
                            result.to_pickle(fname)

                        if correct == self.n_sig:
                            for M in range(M, 11):
                                result[prec][N, M] = correct
                            result.to_pickle(fname)
                            break
                        sys.stderr = stderr
                        

    def make_prec_table(self):

        fname = f"../data/prec_n={self.n_sig}_L={self.upper_int}"

        precs = [23, 53]

        try:

            open(fname)

        except FileNotFoundError:
            Ns = range(1, 101)
            
            table = {prec: {
                (N, M): np.nan for N in Ns for M in range(1, 11)
            } for prec in precs}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname)

        print(table)

        M_max = 11

        print(r"\begin{tabular}{cc|%s}" % ((M_max - 1) * "c"))
        header = f"Precision & N"
        for M in range(1, M_max):
            header += f" & {M}"
        print(f"{header} \\\\")
        print(r"\midrule\midrule")
        for prec, name in zip(precs, ["Single", "Double"]):

            row = f"{name}"
            for N in self.Ns:
                row += f" & {N}"

                for M in range(1, min(reduced(N) // 2, M_max)):
                    row += f" & {table[prec][N, M] / self.n_sig * 100:.1f}"
                for M in range(reduced(N) // 2, M_max):
                    row += r" & \textbar"
                
                print(f"{row} \\\\")
                row = ""

            print(r"\midrule")
        print(r"\end{tabular}")


    def test_L(self, step=1e2, n_steps=30, N=30, targets=[100]):

        fname = f"../data/L_N={N}_n={self.n_sig}"
        results = pd.read_pickle(fname)

        targets = targets if targets else [50, 90, 100]

        Ls = results[1]["L"]
        signals = {L: self.rand.binomial(L, .5, (self.n_sig, N)) for L in Ls}

        for M in range(3, 4):
            print(f"M = {M}")
            for target in targets:
                print(f"target = {target}")
                data = []
                for L in results[1]["L"]:

                    with mp.get_context() as context:
                        print(f"L = {L}")
                        context.precision = 400

                        known_coeffs = shallow_select_coeffs(N, M)
                        blurred_signals = binvert.sample_1D(signals[L], known_coeffs)

                        with self.subTest(N=N, M=M):

                            def key(beta2):
                                correct = 0
                                for n in range(self.n_sig):
                                    try:
                                        inverted = binvert.invert_1D(blurred_signals[n], known_coeffs, beta2=beta2)
                                        correct += np.allclose(inverted, signals[L][n])
                                    except:
                                        pass
                                    
                                return correct / self.n_sig

                            percentile, value = exp_search(step, target / 100, key, n_steps)
                            data.append(percentile)

                            print(f"M={M}, {target}th percentile: {percentile:.2e} (value: {value:.2f})")

                            stderr = sys.stderr
                            with open("/dev/null", "w") as f:
                                sys.stderr = f
                                results[M].update({target: data})
                                results.to_pickle(fname)
                            sys.stderr = stderr

                print("Current results:")
                print(results)

    def make_L_plot(self, N=31):

        fname = f"../data/L_N={N}_n={self.n_sig}"

        n_Ls = 8
        targets = [50, 90, 100]
        Ls = np.arange(1, n_Ls + 1) ** 2

        try:

            open(fname)
 
        except FileNotFoundError:

            table = {M: 
                     {"L": Ls, "theory": np.array([theoretical_beta2(N, M, L) for L in Ls])} | {target: np.zeros(n_Ls) for target in targets}
                    for M in range(1, 4)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname) 

        print(table)

        n_M = 3
        fig, ax = plt.subplots(1, n_M)
        fig.set_size_inches(12, 5)

        for M in range(1, n_M + 1):
            
            ca = ax[M - 1]
            ca.plot(Ls, table[M]["theory"], label="Theory")
            for target in targets:
                ca.plot(Ls, table[M][target], label=f"{target}th")
            ca.set_xlabel(r"$L$")
            ca.set_ylabel(r"$\beta_2$")

            ca.set_yscale('log')
            ca.set_title(r"$M = %d$" % M)

            # ca.set_ylim(0, 1e5)

        ax[-1].legend()

        plt.subplots_adjust(wspace=0.6)

        save_fig("L_plot")
        plt.show()

            
    def make_full_precision_plot(self, N = 19):

        precs = [0, 5, 10] # number of decimal digits

        fname = f"../data/full_prec_n={self.n_sig}"

        try:

            open(fname)

        except FileNotFoundError:

            table = {(N, prec): {
                "L": np.arange(int(N ** .5), N // 2 + 1),
                "n_rec": np.nan,
                "n_dir": np.nan
                } for N in self.Ns for prec in precs}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname) 

        print(table)

        fig, ax = plt.subplots(1, 2)

        Ls = np.arange(int(N ** .5), N // 2 + 1)

        for i, var in enumerate(["n_rec", "n_dir"]):

            for prec in precs:
                # print(table[N, prec])
                results = pd.DataFrame(table[N, prec].to_dict())
                print(results)
                # Ls = results["L"]
                data = np.concatenate([results[var] / self.n_sig / (1 + N * i), np.ones(len(Ls) - len(results["L"]))])
                ax[i].plot(Ls, data, label=f"prec={prec}")
            ax[i].legend()

            ax[i].set_title(["Percentage of test matrices recovered", "Percentage of directions recovered"][i])

        plt.show()


class Matrix(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(86547689)

        self.n_sig = 100

    def test_rect(self, N1=30, N2=40):

        fname = f"../data/rect_n={self.n_sig}"
        results = pd.read_pickle(fname)

        signals = self.rand.integers(0, 2, (self.n_sig, N1, N2))
        print(f"N1 x N2 = {N1} x {N2}")

        for M in range(1, 10):
            print(f"M = {M}")
            # known_coeffs = binvert.select_coeffs_2D(N1, N2, [M] * sp.divisor_count(N1 * N2))
            known_coeffs = binvert.select_coeffs_2D(N1, N2, [M])
            n_coeffs = int(sum(map(lambda set: sum(map(len, set)), known_coeffs.values())))
            blurred = binvert.sample_2D(signals, known_coeffs)

            correct = 0
            stime = time()
            for n in range(self.n_sig):

                try:
                    inverted = binvert.invert_2D(blurred[n], known_coeffs, beta2=1e14, epsilon=1e-10)
                    correct += np.allclose(inverted - signals[n], 0)
                except:
                    pass
            
            stderr = sys.stderr
            with open("/dev/null", "w") as f:
                sys.stderr = f
                results[N1, N2, M].update({"n_rec": correct, "time": time() - stime, "n_coeffs": n_coeffs})
                results.to_pickle(fname)
            sys.stderr = stderr

            if correct == self.n_sig:
                break

    def test_all_rect(self):

        Ns = [(4, 6), (8, 10), (9, 11), (11, 13), (12, 18), (9, 21), (30, 40), (36, 45), (29, 29), (31, 31), (37, 37), (41, 41), (49, 49), (60, 60), (23, 23)]
        last = np.load("../data/rect_last.npy")

        for i in range(last + 1, len(Ns)):
            N1, N2 = Ns[i]

            self.test_rect(N1, N2)
            np.save("../data/rect_last.npy", i)

    
    def make_rect_table(self):

        fname = f"../data/rect_n={self.n_sig}"

        Ns = product(range(2, 51), repeat=2)
        try:

            open(fname)

        except FileNotFoundError:

            table = {(N1, N2, M): {
                "n_rec": -1,
                "time": np.nan,
                "n_coeffs": -1,
                } for N1, N2 in Ns for M in range(10)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

            np.save("../data/rect_last.npy", -1)

        finally:

            table = pd.read_pickle(fname) 

        print(table)
        
        M_max = 9
        Ns = [(4, 6), (8, 10), (9, 11), (11, 13), (12, 18), (9, 21), (30, 40), (36, 45)]

        print()

        print(r"\begin{tabular}{ccc|cccc}")

        print(r"$N_1 \times N_2$ & $N$ & M & Coeffs, \# & $\beta_2$ & Rec, \% & $t$, sec \\")

        print("\midrule\midrule")

        for N1, N2 in Ns:
            
            print(f"${N1} \\times {N2}$ & ${np.lcm(N1, N2)}$", end='')

            for M in range(1, M_max + 1):

                n_rec = table[N1, N2, M]["n_rec"]
                time = table[N1, N2, M]["time"]
                n_coeffs = table[N1, N2, M]["n_coeffs"]
                beta = theoretical_beta2(np.lcm(N1, N2), M, np.gcd(N1, N2))

                if M != 1:
                    print("&", end='')
                print(f"& {M} & {int(n_coeffs)} & {float(beta):.2e} & {n_rec / self.n_sig: .2f} & {time/self.n_sig: .2f} \\\\")

                if n_rec == self.n_sig:
                    break

            print("\midrule")

        print("\\end{tabular}")

    def make_image_plot(self, N=60, M=[1], L=13, check=True):
        # tested binary N = 70, 80, 90, 100 with M = 1
        # double precision: N=60, L=13
        # extra precision: N=60, L=230

        image = Image.open("../Set12/01.png")
        image = np.array(image).astype(np.int64)

        plots = [image]
        # plots = []

        def resize(image, size):
            N, N = image.shape
            samples = np.arange(0, N, N // size)[:size]
            return image[samples][:, samples]

        def rescale(image, range):
            if range is None:
                return image
            vmin, vmax = np.min(image), np.max(image)
            rescaled = image / (vmax - vmin)
            return np.round(rescaled * range).astype(int)

        resized = skimage.transform.resize(image, (N, N))
        rescaled = rescale(resized, L)
        # plots.append(rescaled)

        resized = resize(image, N)
        rescaled = rescale(resized, L)

        print(len(np.unique(rescaled)))

        plots.append(rescaled)

        print(f"L = {np.max(rescaled)}")
        print(f"Theoretical beta2: {theoretical_beta2(N, M[0], N*np.max(rescaled)):.2e}")

        known_coeffs = binvert.select_coeffs_2D(N, N, M)
        blurred = binvert.sample_2D(rescaled, known_coeffs)

        print("Number of coefficients required for recovery:", sum(map(len, known_coeffs.values())))
        print("Number of coefficients used for recovery:", sum(map(lambda set: sum(map(len, set)), known_coeffs.values())))

        plots.append(blurred)

        if check:

            try:
                inverted = binvert.invert_2D(blurred, known_coeffs, beta2=1e14, epsilon=1e-8)
                assert np.allclose(inverted - rescaled, 0)
            except:
                print("double precision insufficient")
                with mp.get_context() as c:
                    c.precision = 200
                    blurred = binvert.sample_2D(rescaled, known_coeffs)
                    inverted = binvert.invert_2D(blurred, known_coeffs, beta2=1e30, epsilon=1e-10)
                    self.assertTrue(np.allclose(inverted - rescaled, 0), "Higher precision test")
        else:
            print("did not check if this setup was solvable")

        fig, ax = plt.subplots(2, len(plots))
        fig.set_size_inches(12, 5)
        for i in range(len(plots)):
            ax[0, i].matshow(-plots[i], cmap='binary')
            ax[0, i].set_xticks([])
            ax[0, i].set_yticks([])
        ax[0, 0].set_title("Original image")
        ax[0, 1].set_title("Rescaled image")
        ax[0, 2].set_title("Sampled image")

        ax[1, 0].set_title("Value distribution")
        kde = scipy.stats.gaussian_kde(image.flatten())
        x = np.linspace(np.min(image), np.max(image), 1000)
        ax[1, 0].plot(x, kde(x))

        ax[1, 1].set_title("Value distribution")
        kde = scipy.stats.gaussian_kde(rescaled.flatten())
        x = np.linspace(np.min(rescaled), np.max(rescaled), 1000)
        ax[1, 1].plot(x, kde(x))
        
        plt.show()







class Misc(unittest.TestCase):

    def draw_subgroup_lattice_1D(self, M=30):

        print(r"\begin{tikzpicture}")

        print(r"\foreach \n/\x/\y/\group in {")
        divisors = {M}
        layer = 0
        connections = set()
        groups = {}

        while divisors:
            new_divisors = set()
            ns = []
            for d in divisors:
                groups[d] = set(sorted(n % M for n in range(0, M, d) if np.gcd(n, M) == d))
                ns.append(d)
                for p in sp.primefactors(d):
                    new_divisors.add(d // p)
                    connections.add((d, d // p))
            print(f"% layer {layer}")
            stretch = max(max(map(len, groups.values())) / 5, 1)
            for i, n in enumerate(ns):
                print(f"{n}/{stretch * (2 * i - (len(ns) - 1))}/{layer}/{groups[n]},")

            divisors = new_divisors
            layer += 1

        print(r"} { \node (\n) at (\x,\y) {\set\group}; }")

        print(r"  \foreach \a/\b in {")
        for d1, d2 in connections:
            print(f"{d1}/{d2},")
        print(r"  } { \draw (\a) -- (\b); }")
        
        print(r"\end{tikzpicture}")

        
    def draw_subgroup_lattice_2D(self, M=4, N=6, color=True, cyclic=True):

        assert cyclic

        subgroups = []
        found = np.zeros((M, N), dtype=bool)
        lcm = np.lcm(M, N)
        for k, l in product(range(M), range(N)):
            if found[k, l]: continue

            subgroup = frozenset((k * lam % M, l * lam % N) for lam in range(lcm))
            generating_set = frozenset((k * lam % M, l * lam % N) for lam in range(lcm) if np.gcd(lam, lcm) == 1)

            for k, l in generating_set:
                found[k, l] = True

            subgroups.append((subgroup, generating_set))

        n_sets = len(subgroups)
        
        trivial = frozenset({(0, 0)})
        trivial_ind = subgroups.index((trivial, trivial))
        levels = {0: {trivial_ind}}
        remaining = set(range(n_sets)) - {trivial_ind}

        connections = set()

        level = 0
        while remaining:
            levels[level + 1] = set()
            # print(f"Level: {level}; remaining: {remaining}")
            for subgroup_ind in remaining:
                subgroup, generating_set = subgroups[subgroup_ind]
                for subsubgroup_ind in levels[level]:
                    subsubgroup, _ = subgroups[subsubgroup_ind]
                    # print(subsubgroup, subsubgroup <= subgroup)
                    if subsubgroup <= subgroup and all(not (subgroup >= subgroups[ind][0] >= subsubgroup) for ind in remaining - {subgroup_ind}):
                        levels[level + 1].add(subgroup_ind)
                        connections.add((subgroup_ind, subsubgroup_ind))
            level += 1
            remaining -= levels[level]
        
        print()
        print(r"\begin{tikzpicture}")

        print(r"\foreach \n/\x/\y/\group in {")
        # print(r"\foreach \n/\x/\y/\group/\div1/\div2 in {")
        for level in range(len(levels)):
            print(f"% level {level}")
            count = len(levels[level]) - 1
            scale = 1.5 * max(map(lambda index: len(subgroups[index][1]), levels[level]))
            # scale = 2 * max(map(lambda index: len(subgroups[index][1]), levels[level]))
            i = 0
            for subgroup_ind in levels[level]:
                subgroup, generating_set = subgroups[subgroup_ind]
                x = scale * (i - count / 2)
                if count > 6:
                    y = 2 * level + .5 * (i % 2)
                    x //= 2
                else:
                    y = 2 * level
                i += 1
                k, l = list(generating_set)[0]
                print(f"{subgroup_ind}/{x}/{y}/{set(sorted(list(generating_set)))}", end='%\n' if i == len(levels[level]) and level == len(levels) - 1 else ',\n')
                # print(f"{subgroup_ind}/{x}/{y}/{set(sorted(list(generating_set)))}/{M//np.gcd(M,k)},{N//np.gcd(N,l)}", end='%\n' if i == len(levels[level]) and level == len(levels) - 1 else ',\n')
        print(r"} { \node (\n) at (\x,\y) {\set\group}; }")
        # print(r"} { \node (\n) at (\x,\y) {(\div1,\div2)~,~\set\group}; }")

        colors = ["red", "green", "blue", "cyan", "magenta", "yellow"]
        divisors = sp.divisors(M)
        if color:
            cmap = dict(zip(divisors, colors))
        else:
            cmap = dict(zip(divisors, ["black"] * len(divisors)))
        print(r"\foreach \a/\b/\color in {")
        for i, (d1, d2) in enumerate(connections):
            subgroup, generating_set = subgroups[d1]
            generator = list(generating_set)[0]
            color = cmap[np.gcd(generator[0], M)]
            print(f"{d1}/{d2}/{color}", end=',\n' if i < len(connections) - 1 else '%\n')
        print(r"} { \draw[color=\color] (\a) -- (\b); }")
        
        print(r"\end{tikzpicture}")
        

    def search_space(self, M = 30, N = 1, L = 1):
        
        all_coeff_classes = binvert.get_coeff_classes_2D(M, N, False)
        n_coeffs = sum(map(len, all_coeff_classes.values()))
        print(f"Number of coefficient classes: {n_coeffs}")

        total_size = 0.0

        for D1, D2 in product(sp.divisors(M), sp.divisors(N)):

            D = np.lcm(D1, D2)
            D_red = reduced(D)

            L1 = L * M * N // D

            pow = D_red - 2
            if pow > 0:
                subproblem_size = (L1 + 1) ** pow
                total_size += subproblem_size * len(all_coeff_classes[D1, D2])
                print(D1, D2, D, subproblem_size, len(all_coeff_classes[D1, D2]))
                
        print(f"Total search space size: {total_size:.2e}")

        pow = (M * N - 2 * n_coeffs + 2 + (M * N) % 2)
        print(f"Brute force search space size: 2^{pow} = {(L + 1) ** pow:.2e}")