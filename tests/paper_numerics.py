import numpy as np
import scipy
import gmpy2 as mp
import sympy as sp
import sys
import pandas as pd
import unittest
import skimage
import pickle
from PIL import Image
from itertools import product, chain
from time import time

from context import intvert

import tracemalloc

import matplotlib.pyplot as plt
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def vol(n):

    return np.pi ** (n / 2) / sp.functions.special.gamma_functions.gamma(n / 2 + 1)

def theoretical_beta2(N, M, upper_int, beta0=.1, K=None,):

    N_reduced = reduced(N)

    if K is None:
        K = .5 * ((N_reduced - (2 if N > 2 else 1) * M) * upper_int) ** .5

    first_term = vol(N_reduced) * (K ** 2 + beta0 ** 2) ** ((N_reduced + M) / 2) * ((N_reduced + 1) * vol(N_reduced) / (2 * N_reduced * vol(N_reduced - 1)))
    second_term = vol(N_reduced) * K ** (N_reduced + M) * sum(1 / np.arange(1, max(1, np.floor(((K / beta0) ** 2 + 1) ** .5))) ** M)
    return float(first_term + second_term) ** (1 / (2 * M))

def gmean(x, y):
    return (x * y) ** .5

def exp_search(current, step, target, key, n_max):
    best_lower_bound = 0
    best_upper_bound = 0
    last = 1
    closest = (current, 0)
    for n in range(n_max):
        value = key(current)
        if abs(target - value) < abs(closest[1] - target) or (
            abs(target - value) == abs(closest[1] - target)
            and current < closest[0]
        ):
            closest = current, value

        # print(value, current)
        if value < target:
            best_lower_bound = current
            last = best_upper_bound
            if best_upper_bound:
                current = gmean(best_upper_bound, current)
            else:
                current = step * current
        else:
            best_upper_bound = current
            last = best_upper_bound
            if best_lower_bound:
                current = gmean(best_lower_bound, current)
            else:
                current = step / current
    return closest

def save_fig(fname):
    plt.savefig(f"../figures/{fname}.png", bbox_inches='tight')
    
def shallow_select_coeffs(N, M = 1):

    prime_factors = sp.factorint(N)
    if N in prime_factors:
        Ls = [M]
    else:
        Ls = [M] + [N] * sum(prime_factors.values())
    
    return intvert.select_coeffs_1D(N, Ls)

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
            blurred_signals = intvert.sample_1D(signals, selected_coeffs)
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

    def test_M_N_L(self):


        fig, ax = plt.subplots(1, 3)
        fig.set_size_inches(12, 5)


        Ns = np.arange(29, 35)
        ca = ax[0]
        for i, N in enumerate(Ns):
            # L = L if L else N
            L = 1
            betas = []
            Ms = np.arange(1, (reduced(N)) // 2)
            for M in Ms:
                betas.append(theoretical_beta2(N, M, L))
            ca.plot(Ms, betas, label=f"$N = {N}$")

            ca.set_xlabel(f"$M$")
            ca.set_ylabel(r"$\beta_2$")
            ca.set_yscale('log')

            ca.legend()


        Ns = np.arange(19, 61)
        Ms = np.arange(1, 4)

        ca = ax[1]

        for M in Ms:
            beta2s = []
            for N in Ns:
                L = 1
                beta2s.append(theoretical_beta2(N, M, L))
            ca.plot(Ns, beta2s, label=f"$M = {M}$")
            ca.set_ylabel(r"$\beta_2$")
            ca.set_xlabel(r"$N$")
            ca.set_yscale('log')

            ca.legend()

        M = 1
        ca = ax[2]
        for L in [0, 1, 2]:
            beta2s = []
            for N in Ns:
                beta2s.append(theoretical_beta2(N, M, N ** L))
            ca.plot(Ns, beta2s, label=f"$L = N^{L}$")
            ca.set_ylabel(r"$\beta_2$")
            ca.set_xlabel(r"$N$")
            ca.set_yscale('log')
            ca.legend()

        plt.subplots_adjust(hspace=0.6)
        save_fig("M_N_L_dep")
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
        Ms = np.arange(1, 4)

        fig, ax = plt.subplots()
        xmax = 10
        beta0s = np.concatenate([np.geomspace(1e-5, 1e-1), np.linspace(1e-1, xmax)])
        labels = []
        xticks = []
        for M in Ms:
            labels.append(f"$K_{M}$")
            xticks.append(.5 * ((reduced(N) - 2 * M) * N) ** .5)

            beta2s = []
            for beta0 in beta0s:
                beta2s.append(theoretical_beta2(N, M, N, beta0))

            ax.plot(beta0s, beta2s, label=f"$M = {M}$", color=colors[M - 1])

        ticks = list(np.linspace(0, xmax, 3))
        ax.set_xticks(np.array(xticks + ticks, dtype=float), labels + ticks)
        x_tick_labels = ax.get_xticklabels()

        for i in range(3):
            x_tick_labels[i].set_color(colors[i])
            x_tick_labels[i + 3].set_color("black")

        plt.xlabel(r"$\beta_0$")
        plt.ylabel(r"$\beta_2$")
        plt.yscale('log')
        plt.legend()

        save_fig("beta0_dep")
        plt.show()


    def test_rho(self):

        N = 11
        Ms = np.arange(1, 6)
        gammas = np.arange(0, 10)
        beta0, beta2 = .1, 1e0
        
        for M in Ms:
            beta2 = theoretical_beta2(N, M, N, beta0)
            K = .5 * ((reduced(N) - 2 * M) * N) ** .5
            rhos = np.zeros(gammas.shape)
            rhos[0] = vol(reduced(N)) * (K ** 2 + beta0 ** 2) ** ((reduced(N) + M) / 2) * ((reduced(N) + 1) * vol(reduced(N)) / 2 / reduced(N) / vol(reduced(N) - 1) / beta2 ** 2) ** M
            rhos[1:] = 2 * vol(reduced(N)) * K ** reduced(N) * (K / gammas[1:] / beta2 ** 2) ** M

            if M == reduced(N) // 2:
                rhos[:] = 0
                rhos[1] = 2

            print(np.sum(rhos))
            plt.plot(gammas, rhos, label=f"$M = {M}$")

        plt.xlabel(r"$\gamma$")
        plt.ylabel(r"$\rho$")
        plt.legend()
        save_fig("rho_gamma-theor")
        plt.show()
            
            

class Lattice(unittest.TestCase):

    def setUp(self):
        
        # self.rand = np.random.default_rng(98708743)
        self.rand = np.random.default_rng(9870873)

        self.n_sig = 100
        self.Ns = [19, 24, 25, 27, 30, 31, 32, 39, 45, 46, 47, 48, 49, 50, 60, 90]
        self.upper_int = None

        
    def test_beta_hist(self, n_betas = 40, N = 23, M=None):

        upper_int = self.upper_int if self.upper_int else N
        fname = f"../data/beta_hist_nsig={self.n_sig}_nbeta={n_betas}_L={self.upper_int}"
        results = pd.read_pickle(fname)

        with mp.get_context() as context:

            # context.precision = 400

            M = M if M else range(1, int(np.sqrt(N)))

            with self.subTest(N=N):

                signals = self.rand.binomial(upper_int, .5, (10 * self.n_sig, N))

                for M in M:
                    correct = set()

                    selected_coeffs = shallow_select_coeffs(N, M)
                    print(selected_coeffs)
                    blurred = intvert.sample_1D(signals, selected_coeffs)

                    with self.subTest(M=M):

                        mag_guess = int(np.log10(float(theoretical_beta2(N, M, upper_int))))
                            
                        beta2s = np.logspace(mag_guess * 4 / 7, mag_guess * 4 / 3, n_betas)
                        
                        n_correct = []
                        for n_beta in range(n_betas):
                            n_correct.append(0)

                            beta2 = beta2s[n_beta]

                            for n in range(len(signals)):
                                if n in correct:
                                    n_correct[-1] += 1
                                    continue
                                try:
                                    inverted = intvert.invert_1D(blurred[n], selected_coeffs, beta2=beta2)
                                    n_correct[-1] += np.allclose(signals[n] - inverted, 0)
                                    correct.add(n)
                                except:
                                    pass

                            results[(N, M)].loc["beta"] = beta2s
                            results[(N, M)].loc["correct"] = n_correct
                            results.to_pickle(fname)

                        print(f"N = {N}, M = {M}, correct: {n_correct}")
    
    def make_beta_hist(self, n_betas=40, N=23):

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
            betas = data['beta']
            curve = np.array(data['correct']) / (self.n_sig)

            ax[M - 1].plot(betas, curve)

            ax[M - 1].set_xscale("log")
            ax[M - 1].set_xlabel(r"$\beta_2$")
            ax[M - 1].set_xlim(min(betas), max(betas))
            
            ax[M - 1].set_title(r"$M = %d$" % M)

            theory_beta2 = theoretical_beta2(N, M, upper_int)
            for i in range(n_betas):
                if betas[i] <= theory_beta2 <= betas[i + 1]:
                    break
            height = curve[i] + (curve[i + 1] - curve[i]) / (betas[i + 1] - betas[i]) * (betas[i + 1] - betas[i])
            ax[M - 1].vlines(theory_beta2, 0, height, color='red', linestyle='dashed', label="Theoretical $\\beta_2$")
        
        # plt.suptitle(r"$N = %d$" % N)
        plt.subplots_adjust(wspace=0.6)
        save_fig("beta_hist")
        plt.show()

    def test_percentiles(self, step = 1e2, n_steps = 30, targets = None, Ms=[], upper_int=None):

        fname = f"../data/percentiles_n={self.n_sig}_L={self.upper_int}"
        for N in [90]:#self.Ns:
            # upper_int = upper_int if upper_int else N
            upper_int = N

            with mp.get_context() as context:

                results = pd.read_pickle(fname)

                context.precision = 200

                signals = self.rand.binomial(upper_int, .5, (self.n_sig, N))

                for M in Ms if Ms else range(1, min(4, reduced(N) // 2)):

                    smallest_working = np.full(self.n_sig, np.inf, dtype=object)
                    largest_failing = np.full(self.n_sig, 0, dtype=object)

                    known_coeffs = shallow_select_coeffs(N, M)
                    blurred_signals = intvert.sample_1D(signals, known_coeffs)

                    with self.subTest(N=N, M=M):

                        def key(beta2):
                            correct = 0
                            for n in range(self.n_sig):
                                if largest_failing[n] >= beta2:
                                    continue
                                if smallest_working[n] <= beta2:
                                    correct += 1
                                    continue
                                try:
                                    inverted = intvert.invert_1D(blurred_signals[n], known_coeffs, beta2=beta2, epsilon=1e-10)
                                    if np.allclose(inverted, signals[n]):
                                        correct += 1
                                        smallest_working[n] = beta2
                                    else:
                                        largest_failing[n] = beta2
                                except intvert.InversionError:
                                    largest_failing[n] = beta2
                                
                            return correct / self.n_sig

                        def oldkey(beta2):
                            correct = 0
                            for n in range(self.n_sig):
                                try:
                                    inverted = intvert.invert_1D(blurred_signals[n], known_coeffs, beta2=beta2, epsilon=1e-10)
                                    correct += np.allclose(inverted, signals[n])
                                except intvert.InversionError:
                                    pass
                                
                            return correct / self.n_sig

                        percentiles = []
                        targets = targets if targets else [50, 90, 100]
                        guess = theoretical_beta2(N, M, upper_int)
                        for target in targets:

                            percentile, value = exp_search(target * guess / 100, step, target / 100, key, n_steps)
                            percentiles.append(percentile)

                            print(f"{N}x{N}, M={M}, {target:3}th percentile: {percentile:.2e} (value: {value:.2f}) (actual value: {oldkey(percentile)})")
                            
                            stderr = sys.stderr
                            with open("/dev/null", "w") as f:
                                sys.stderr = f
                                results[(N, M)].update({target: percentile})
                                results.to_pickle(fname)
                            sys.stderr = stderr

                        print(f"{N}x{N}, M={M}, Theoretical: {guess:.2e} (value: {key(guess):.2f}) (actual value: {oldkey(guess):.2})")

    
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
            header += f" & {target}th percentile $\\beta_2$"
        print(header + " \\\\")

        print("\midrule")

        Ns = self.Ns

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
        
        signals = self.rand.binomial(20, .5, (self.n_sig, 38))
        decimated = signals[:, ::2] + signals[:, 1::2]
        
        for N, signals in zip([19, 38], [decimated, signals]):
            L = results[N, 1].loc['L']
            print(f"Testing N = {N}, L = {L}")
            self.test_percentiles(N=N, Ms=[1], signals=signals)

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


    def test_K(self, N=[90]):

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

                signals = self.rand.binomial(upper_int, .5, (self.n_sig, N))

                for  M in range(1, min(5, reduced(N) // 2)):
                    # blurred = blur.blur(matrices, l)

                    selected_coeffs = shallow_select_coeffs(N, M)
                    blurred = intvert.sample_1D(signals, selected_coeffs)

                    n_correct = 0
                    for n in range(self.n_sig):
                        K = np.linalg.norm(signals[n] - blurred[n].astype(float))

                        try: 
                            beta2 = theoretical_beta2(N, M, upper_int, K=K)
                            inverted = intvert.invert_1D(blurred[n], selected_coeffs, beta2=beta2)
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
        for N in range(45, 51):
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

        beta2 = {24: 1e7, 53: 1e14}
        
        for N in Ns:
            upper_int = self.upper_int if self.upper_int else N
            signals = self.rand.binomial(upper_int, .5, (self.n_sig, N))

            # for prec in [24, 53]:
            for prec in [24]:

                for M in range(1, 11):

                    with mp.get_context() as c:
                        c.precision = prec

                        known_coeffs = shallow_select_coeffs(N, M)
                        blurred = intvert.sample_1D(signals, known_coeffs)

                        correct = 0
                        for n in range(self.n_sig):

                            try:
                                inverted = intvert.invert_1D(blurred[n], known_coeffs, beta2=beta2[prec])
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

        precs = [24, 53]

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

        print()
        print(r"\begin{tabular}{cc|%s}" % ((M_max - 1) * "c"))
        header = f"Precision & \slashbox{{$N$}}{{$M$}}"
        for M in range(1, M_max):
            header += f" & {M}"
        print(f"{header} \\\\")
        print(r"\midrule\midrule")
        for prec, name in zip(precs, ["Single", "Double"]):

            row = f"{name}"
            for N in self.Ns:
                row += f"& {N}"

                for M in range(1, min(reduced(N) // 2, M_max)):
                    row += f" & {table[prec][N, M] / self.n_sig * 100:.1f}"
                for M in range(reduced(N) // 2, M_max):
                    row += r" & \textbar"
                
                print(f"{row} \\\\")
                row = ""

            print(r"\midrule")
        print(r"\end{tabular}")
        print()

    def test_prec_plot(self, N=30, n_betas=50):

        n_sig = 10 * self.n_sig
        signals = self.rand.binomial(N, 0.5, (n_sig, N))
        with mp.get_context() as c:
            c.precision = 24
            coeffs = shallow_select_coeffs(N, 1)
            blurred = intvert.sample_1D(signals, coeffs)

            def key(beta2):

                correct = 0
                for n in range(n_sig):

                    try:
                        inverted = intvert.invert_1D(blurred[n], known_coeffs=coeffs, beta2=beta2)
                        correct += np.allclose(inverted, signals[n])
                    except intvert.InversionError:
                        pass
                
                print(f"beta2 = {beta2:.2e}; n_correct = {correct}")
                return correct

            betas = np.logspace(0, 10, n_betas)
            correct = []
            for n in range(n_betas):
                print(f"n = {n}; ", end='')
                beta2 = betas[n]
                correct.append(key(beta2))

            df = pd.DataFrame({"beta2":betas, "correct":correct})
            df.to_pickle(f"../data/prec_n={self.n_sig})_N={N}_n_beta={N}")
                
    def make_prec_plot(self, N=24, n_betas=50):

        fname = f"../data/prec_n={self.n_sig})_N={N}_n_beta={N}"

        try: 
            
            table = pd.read_pickle(fname)
        
        except FileNotFoundError:

            return
        
        betas = table["beta2"]
        correct = table["correct"] / np.max(table["correct"])

        plt.plot(betas, correct)
        plt.xscale('log')
        plt.vlines(theoretical_beta2(N, 1, N), 0, 1, color='red', linestyles='dashed')
        save_fig(f"prec_N={N}")
        plt.show()
            


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
                        blurred_signals = intvert.sample_1D(signals[L], known_coeffs)

                        with self.subTest(N=N, M=M):

                            def key(beta2):
                                correct = 0
                                for n in range(self.n_sig):
                                    try:
                                        inverted = intvert.invert_1D(blurred_signals[n], known_coeffs, beta2=beta2)
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


    def test_LLL(self, N=47, M=1, n_betas=19):

        beta0, beta1, beta3, delta = 1e-1, 1e10, 1e2, .9972

        n_sig = 3 * self.n_sig
        solved = np.zeros(n_sig, dtype=bool)
        not_shorter = np.zeros(n_sig, dtype=bool)
        def test(beta2):
            with mp.get_context() as c:
                c.precision = 200

                signals = self.rand.binomial(N, 0.5, (n_sig, N))

                from intvert.invert import _approximate_svp, _get_basis_matrix
                factors = sp.primefactors(N)
                # for M in range(1, min(reduced(N) // 2, 4)):
                known_coeffs = shallow_select_coeffs(N, M)
                blurred = intvert.sample_1D(signals, known_coeffs)

                n_shorter = 0
                n_correct = 0

                for n in range(n_sig):

                    # if not_shorter[n] and solved[n]:
                    #     n_correct += 1
                    #     continue

                    signal = signals[n]
                    K = None

                    dft = intvert.mp_dft(signal)
                    decimated = {}
                    for p in factors:
                        decimated[N // p] = np.round(np.fft.ifft(dft[::p].astype(complex)).real).astype(int)

                    basis_matrix = _get_basis_matrix(blurred[n], dft, decimated, known_coeffs[1], factors)

                    scaled_basis_matrix = np.copy(basis_matrix)
                    scaled_basis_matrix[N] *= beta0
                    scaled_basis_matrix[N + 1: -2 * M] *= beta1
                    scaled_basis_matrix[-2 * M:] *= beta2
                    scaled_basis_matrix *= beta3

                    reduced_basis = _approximate_svp(np.vectorize(int, otypes = [object])(scaled_basis_matrix).transpose(), delta) / beta3

                    actual_shortest = reduced_basis[0]
                    expected_shortest = (scaled_basis_matrix / beta3 @ np.concatenate([signal, [1]])).astype(float)
                    shorter = (np.linalg.norm(actual_shortest - expected_shortest, ord=np.inf) >= 1 
                        and np.linalg.norm(actual_shortest + expected_shortest, ord=np.inf) >= 1 
                        and np.linalg.norm(actual_shortest) < np.linalg.norm(expected_shortest) - .1)
                    
                    def check():
                        for vector in reduced_basis:
                            for sign in [-1, 1]:
                                if np.allclose(intvert.mp_round(sign * vector[:N] + blurred[n]), signal):
                                    return True
                        return False
                    # solved[n] |= check()
                    solved[n] = check()
                    shorter = shorter and not solved[n]
                    # if not_shorter[n]:
                    #     shorter = False
                    # else:
                    #     not_shorter[n] = not shorter                    

                    n_correct += solved[n]
                    n_shorter += shorter

                print(f"N = {N}; M = {M}; n_correct = {n_correct:3}; n_shorter = {n_shorter:3}")
                return n_correct, n_shorter

                    # stderr = sys.stderr
                    # with open("devnull", "w") as file:
                    #     sys.stderr = file
                    #     table[M][table[M]["N"] == N]["n_correct"] = n_correct
                    #     table[M][table[M]["N"] == N]["n_shorter"] = n_shorter
                    #     table[M][table[M]["N"] == N]["n_both"] = n_both
                    # sys.stderr = stderr
                    # with open(fname, "wb") as file:
                    #     pickle.dump(table, file)

        guess = theoretical_beta2(N, M, N)
        mag = np.log10(guess)
        # betas = np.logspace(mag * .7, 1.2 * mag, n_betas) # for N = 47
        betas = np.logspace(mag * .5, 1.1 * mag, n_betas) # for N = 19

        corrects = []
        shorters = []
        for beta2 in betas:
            correct, shorter = test(beta2)

            corrects.append(correct)
            shorters.append(shorter)

        fname = f"../data/lll_n={self.n_sig}_N={N}_M={M}_n-beta={n_betas}"
        table = {"betas": betas, "correct": corrects, "shorter": shorters}
        df = pd.DataFrame(table)
        df.to_pickle(fname)
        

    def make_LLL_plot(self, N=47, M=1, n_betas=50):

        fname = f"../data/lll_n={self.n_sig}_N={N}_M={M}_n-beta={n_betas}"

        try:

            table = pd.read_pickle(fname)
 
        except FileNotFoundError:

            return

        betas=table["betas"]
        correct = np.array([max(table["correct"][:i + 1]) for i in range(n_betas)])
        shorter = np.array([min(table["shorter"][:i + 1]) for i in range(n_betas)])
        plt.plot(betas, correct / max(table["correct"]), label="Inverted")
        plt.plot(betas, shorter / max(table["correct"]), label="Too short")
        plt.xscale('log')

        theory_beta2 = theoretical_beta2(N, M, N)
        for i in range(n_betas):
            if betas[i] <= theory_beta2 <= betas[i + 1]:
                break
        height = correct[i] + (correct[i + 1] - correct[i]) / (betas[i + 1] - betas[i]) * (betas[i + 1] - betas[i])
        plt.vlines(theory_beta2, 0, correct[i] / max(table["correct"]), color='red', linestyle='dashed', label="Theoretical $\\beta_2$")
        plt.xlim(min(betas), max(betas))

        save_fig(f"LLL_N={N}")
        plt.legend()
        plt.show()

    def LLL_results(self):

        fname = f"../data/lll_n={self.n_sig}"

        Ns = np.arange(1, 101)
        Ms = np.arange(1, 4)

        try:

            open(fname)
            open("")
 
        except FileNotFoundError:

            table = {M: pd.DataFrame({"N": Ns, "n_correct": np.zeros(Ns.size), "n_shorter": np.zeros(Ns.size), "n_both": np.zeros(Ns.size)}) for M in Ms}

            # table = pd.DataFrame(table)
            with open(fname, "wb") as file:
                pickle.dump(table, file)

        finally:

            with open(fname, "rb") as file:
                table = pickle.load(file)

        print(table)

        Ns = np.array([19, 29, 31, 47])

        for M in Ms:
            df = table[M].iloc[Ns - 1]
            
            print(df)

            
import qrcodegen
def generate_qr(message: str, size: int | None = None, ecl: str | None = None, mask: int = -1):

    version = (size - 17) // 4
    boost_ecl = False
    match ecl:
        case "low":
            eclvl = qrcodegen.QrCode.Ecc.LOW
        case "med":
            eclvl = qrcodegen.QrCode.Ecc.MEDIUM
        case "quart":
            eclvl = qrcodegen.QrCode.Ecc.QUARTILE
        case "high":
            eclvl = qrcodegen.QrCode.Ecc.HIGH
        case None:
            boost_ecl = True
            eclvl = qrcodegen.QrCode.Ecc.LOW
        case _:
            pass

    segs = qrcodegen.QrSegment.make_segments(message)
    
    code = qrcodegen.QrCode.encode_segments(segs = segs, ecl = eclvl, minversion = version, maxversion = version, mask = mask, boostecl = boost_ecl)

    return np.array([[code.get_module(col, row) for col in range(size)] for row in range(size)], dtype = np.int64)

class Matrix(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(86547689)

        self.n_sig = 100

    def test_rect(self, N1=90, N2=90):

        fname = f"../data/rect_n={self.n_sig}"
        results = pd.read_pickle(fname)

        signals = self.rand.integers(0, 2, (self.n_sig, N1, N2))
        print(f"N1 x N2 = {N1} x {N2}")

        for M in range(1, 10):
            print(f"M = {M}")
            # known_coeffs = binvert.select_coeffs_2D(N1, N2, [M] * sp.divisor_count(N1 * N2))
            known_coeffs = intvert.select_coeffs_2D(N1, N2, M)
            n_coeffs = int(sum(map(lambda set: sum(map(len, set)), known_coeffs.values())))
            blurred = intvert.sample_2D(signals, known_coeffs)

            stime = time()
            if M < np.lcm(N1, N2):
                correct = 0
                for n in range(self.n_sig):

                    try:
                        inverted = intvert.invert_2D(blurred[n], known_coeffs, beta2=1e14, epsilon=1e-10)
                        correct += np.allclose(inverted - signals[n], 0)
                    except:
                        pass
            else:
                correct = self.n_sig
            
            stderr = sys.stderr
            with open("/dev/null", "w") as f:
                sys.stderr = f
                results[N1, N2, M].update({"n_rec": correct, "time": time() - stime, "n_coeffs": n_coeffs})
                results.to_pickle(fname)
            sys.stderr = stderr

            if correct == self.n_sig:
                break

    def test_all_rect(self):

        Ns = [(4, 6), (8, 10), (9, 11), (11, 13), (12, 18), (9, 21), (30, 40), (36, 45), (29, 29), (31, 31), (37, 37), (41, 41), (49, 49), (60, 60), (23, 23), (90, 90), (43, 43), (40, 40)]
        last = np.load("../data/rect_last.npy")

        for i in range(last + 1, len(Ns)):
            N1, N2 = Ns[i]

            self.test_rect(N1, N2)
            np.save("../data/rect_last.npy", i)
    
    def make_rect_table(self):

        fname = f"../data/rect_n={self.n_sig}"

        Ns = product(range(2, 91), repeat=2)
        try:

            open(fname)

        except FileNotFoundError:

            table = {(N1, N2, M): {
                "n_rec": -1,
                "time": np.nan,
                "n_coeffs": -1,
                } for N1, N2 in Ns for M in range(reduced(np.lcm(N1, N2)) // 2)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

            np.save("../data/rect_last.npy", -1)

        finally:

            table = pd.read_pickle(fname) 

        print(table)
        
        M_max = 9
        Ns = [(4, 6), (8, 10), (9, 11), (11, 13), (12, 18), (9, 21), (30, 40), (36, 45), 23, 29, 31, 37, 40, 41, 43, 49, 60, 90]

        print()

        print(r"\begin{tabular}{ccl|cccc}")

        print(r"$N_1 \times N_2$ & $N$ & M & Coeffs, \# & $\beta_2$ & Rec, \% & $t$, sec \\")

        print("\midrule\midrule")

        for N in Ns:
            try:
                N1, N2 = N
            except:
                N1, N2 = N, N

            N = np.lcm(N1, N2)
            
            print(f"${N1} \\times {N2}$ & ${N}$", end='')

            for M in range(1, reduced(np.lcm(N1, N2)) // 2):

                n_rec = table[N1, N2, M]["n_rec"]
                time = table[N1, N2, M]["time"]
                n_coeffs = table[N1, N2, M]["n_coeffs"]
                beta = theoretical_beta2(np.lcm(N1, N2), M, np.gcd(N1, N2))

                theory_beta = max(theoretical_beta2(D, M if N1 == 90 and D == 45 else 1, N1 * N2 // D) for D in sp.divisors(N)[:-1]) # largest of subproblems
                theory_beta = max(theoretical_beta2(N, M, N1 * N2 // N), theory_beta)
                if M != 1:
                    print("&", end='')
                print(f"& {M}", end='')
                if M != 1 and N1 == 90:
                    print(r"\textsuperscript{*}", end='')
                print(f" & {int(n_coeffs)} & {float(beta):.2e}/{float(theory_beta):.2e} & {n_rec / self.n_sig: .2f} & {time/self.n_sig: .2f} \\\\")

                if n_rec == self.n_sig:
                    break

            print("\midrule")

        print("\\end{tabular}")

        print(f"\n {theoretical_beta2(45, 1, 90 ** 2 // 45)}")

    def get_image(N, L):
        image = Image.open("../Set12/01.png")
        image = np.array(image).astype(np.int64)
        plt.matshow(-image, cmap='binary')
        save_fig("original")
        # plt.show()
        plt.cla()
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

        # return rescale(resize(image, N), L)
        resized = skimage.transform.resize(image, (N, N), preserve_range=True)
        image = Image.fromarray(resized.astype(np.uint8))
        # print(resized.astype(np.int8))
        image.save(f"../Set12/image1_N={N}.png")
        return rescale(resized, L)

    def make_all_plots(self, check=False):
        # tested binary N = 70, 80, 90, 100 with M = 1
        # double precision: N=60, L=13
        # extra precision: N=60, L=230

        M = 1
        for N, prec, L, beta2 in [(60, 53, 13, 1e14), (60, 200, 256, 1e30), (100, 200, 1, 1e30),]:
            print(f"{N} x {N}")
            image = Matrix.get_image(N, L)
            self.make_image_plot(image, N, prec, beta2, M, check)
            
            save_fig(f"image_N={N}_L={L}")
            plt.show()
            return 

        N = 45
        M = 1
        image = generate_qr("Hello World!", N, "high")
        self.make_image_plot(image, N, 53, 1e14, M, check)
        save_fig("qr")
        plt.show()
        
        return
        
        N = 30
        prec, beta2 = 19, 4e4
        N = 20
        prec, beta2 = 16, 3e3
        image = self.rand.integers(0, 2, (N, N))
        self.make_image_plot(image, N, prec, beta2, 1)

    def make_image_plot(self, image, N, prec, beta2, M, check=True):
        
        print(f"Theoretical beta2: top level: {theoretical_beta2(N, 1, N*np.max(image)):.2e}; subproblem max: {max(theoretical_beta2(d, 1, N**2*np.max(image)//d) for d in sp.divisors(N)[2:]):.2e}")

        known_coeffs = intvert.select_coeffs_2D(N, N, M)

        print("Number of coefficients required for recovery:", sum(map(len, known_coeffs.values())))
        print("Number of coefficients used for recovery:", sum(map(lambda set: sum(map(len, set)), known_coeffs.values())))

        blurred = intvert.sample_2D(image, known_coeffs)
        if check:
            with mp.get_context() as c:
                c.precision = prec
                blurred = intvert.sample_2D(image, known_coeffs)
                # blurred = binvert.sample_2D(image)
                inverted = intvert.invert_2D(blurred, known_coeffs, beta2=beta2)
                # inverted = binvert.invert_2D(blurred, beta2=beta2, epsilon=1e-10)
                self.assertTrue(np.allclose(inverted, image), "Correct Recovery")
        else:
            print("did not check if this setup was solvable")

        plots = [image, blurred.astype(float)]

        fig, ax = plt.subplots(1, len(plots))
        fig.set_size_inches(12, 5)
        for i in range(len(plots)):
            ax[i].matshow(-plots[i], cmap='binary')
            # ax[i].set_xticks([])
            # ax[i].set_yticks([])
        
        ax[0].set_title("Rescaled image")
        ax[1].set_title("Sampled image")

        # plt.cla()

        # fig, ax = plt.subplots(1, 3)

        # ax[0].matshow(-plots[1], cmap='binary')
        # ax[2].matshow(-plots[0], cmap='binary')

        # ax[0].set_xticks([])
        # ax[0].set_yticks([])
        # ax[0].axis('off')
        # ax[2].set_xticks([])
        # ax[2].set_yticks([])
        # ax[2].axis('off')

        # ax[1].arrow(.25, .5, .25, 0, width = .25, head_width = .5, color = 'black', head_length = .25)
        # ax[1].set_ylim(0, 1)
        # ax[1].set_xlim(0, 1)
        # ax[1].set_aspect('equal')
        # ax[1].set_xticks([])
        # ax[1].set_yticks([])
        # ax[1].axis('off')

        plt.show()



class Misc(unittest.TestCase):

    def draw_subgroup_lattice_1D(self, M=30, all_generators=False, subproblems=True):

        print()
        print(r"\begin{tikzpicture}")

        if all_generators:
            print(r"\foreach \n/\x/\y/\group in {")
        elif subproblems:
            print(r"\foreach \n/\x/\y/\size in {")
        else:
            print(r"\foreach \n/\x/\y/\gen in {")

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
                if all_generators:
                    print(f"{n}/{stretch * (2 * i - (len(ns) - 1))}/{layer}/{groups[n]}", end='')
                elif subproblems:
                    print(f"{n}/{stretch * (2 * i - (len(ns) - 1))}/{layer}/{M//(min(groups[n]) if min(groups[n]) else M)}", end='')
                else:
                    print(f"{n}/{stretch * (2 * i - (len(ns) - 1))}/{layer}/{min(groups[n])}", end='')
                if i < len(ns) - 1:
                    print(",")

            divisors = new_divisors
            layer += 1
            if layer <= len(sp.primefactors(M)):
                print(",")
            else:
                print("%")

        if all_generators:
            print(r"} { \node (\n) at (\x,\y) {\set\group}; }")
        elif subproblems:
            print(r"} { \node (\n) at (\x,\y) {${\bf x}^{(\size)}$}; }")
        else:
            print(r"} { \node (\n) at (\x,\y) {$\langle\gen\rangle$}; }")

        print(r"  \foreach \a/\b in {")
        for i, (d1, d2) in enumerate(connections):
            print(f"{d1}/{d2}", end='')
            if i < len(connections) - 1:
                print(",")
            else:
                print("%")
        print(r"  } { \draw (\a) -- (\b); }")
        
        print(r"\end{tikzpicture}")
        print()

        
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
        

    def search_space(self, M = 30, N = 30, L = 1):
        
        all_coeff_classes = intvert.get_coeff_classes_2D(M, N, False)
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

        all_coeffs =  [set(fset).pop() for fset in chain(*intvert.select_coeffs_2D(M, N).values())]
        mat = np.reshape([[[np.exp(-2j * np.pi * (k * m / M + l * n / N)) for m in range(M)] for n in range(N)] for k, l in all_coeffs], (-1, M * N))
        real_mat = np.concatenate([mat.real, mat.imag])
        pow = (M * N - 2 * n_coeffs + 4 - (M * N) % 2 - M % 2 - N % 2)
        print(f"Number of coeff classes: {n_coeffs}")
        assert pow == M * N - np.linalg.matrix_rank(real_mat), f"{pow} != {M * N - np.linalg.matrix_rank(real_mat)}"
        print(f"Brute force search space size: 2^{pow} = {(L + 1) ** pow:.2e}")


    def draw_recursion_tree(self, N = 30):

        level = {0: N}
        
        label = 0
        connections = set()

        max_height = len(sp.factorint(N)) + 2
        width = 0

        print()
        print(r"\begin{tikzpicture}")
        height = 0
        while len(level):
            next_level = {}
            width = max(width, np.log2(max_height - height))
            # print(width)
            for i, node in enumerate(level):
                if height:
                    print(f"\\node ({node}) at ({width * (i - (len(level) - 1) / 2)},{-height}) {{${{\\bf x}}^{{({level[node]})}}$}};")
                else:
                    print(f"\\node ({node}) at (0,0) {{${{\\bf x}}$}};")

                for p in reversed(sp.primefactors(level[node])):
                    label += 1
                    next_level[label] = level[node] // p
                    connections.add((node, label, p))

            level = next_level
            height += 1


        for label1, label2, p in connections:
            # print(f"\\draw ({label1}) -- ({label2}) node[midway,above] {{{p}}};")
            print(f"\\draw ({label1}) -- ({label2});")

        print(r"\end{tikzpicture}")



class Test(unittest.TestCase):

    def one_norm(self):
        """Size of the one-norm bound for gamma=0 case
        """

        def vol(n):

            return np.pi ** (n / 2) / sp.functions.special.gamma_functions.gamma(n / 2 + 1)

        M = 1

        Ns = np.arange(10, 100)
        norms = []
        for N in Ns:
            L = N

            K = .5 * ((reduced(N) - 2 * M) * L) ** .5

            norms.append(2 * reduced(N) * vol(reduced(N) - 1) * K / (reduced(N) + 1) / vol(reduced(N)))


        plt.plot(Ns, norms)
        plt.show()
        
    
    def subproblem(self):
        """Which subproblem has largest value of beta2?
        """

        Ns = np.arange(10, 100)
        M = 2

        for N in Ns:
            L = 1

            divisors = sp.divisors(N)
            beta2s = {d: theoretical_beta2(d, M, L * N // d) for d in divisors if reduced(d) >= 2*M}
            actual = max(beta2s, key=beta2s.get)
            if N % 2:
                    expected = N
            elif N % 4:
                expected = N // 2
            else:
                expected = N
            print(f"N = {N}; expected = {expected:2}; actual = {actual:2}" + ("" if actual == expected else "\t(incorrect)"))

    def integral(self):

        N = 10
        K = 10
        M = 1

        def gap(N=N, r1=1, K=K, n_samples=1000):

            def avg_over_annulus(f, r1, r2):
                samples = np.zeros((n_samples, N))
                for i in range(n_samples):
                    while not(r1 <= np.linalg.norm(x := np.random.uniform(0, r2, N)) <= r2): pass
                    samples[i] = x
                    
                return np.average(np.vectorize(f, signature="(n)->()")(samples))

            if N == 1:
                print("Analytical results:")
                print(2 / (K + r1)) 
                print((np.log(K) - np.log(r1)) / (K - r1))

            def f1(x):
                return np.linalg.norm(x, ord=1)         
            def f2(x):
                return np.linalg.norm(x, ord=1) ** -M
                
            print("Monte Carlo results:")
            res1 = avg_over_annulus(f1, r1, K) ** -M # (1/avg(norm))^M
            res2 = avg_over_annulus(f2, r1, K)       # avg(1/norm^M)
            print(res1)
            print(res2)

            return res2 - res1, (res2 - res1) / res1
        
        gap(N)
        return
        
        Ns = np.arange(1, 11)
        for hole in [1, .001]:
            gaps = []
            for N in Ns:
                abso, rel = gap(N, hole)
                gaps.append(rel)
            plt.plot(Ns, gaps, label=f"hole={hole}")
            print(np.array(gaps))
        plt.xlabel("N")
        plt.ylabel("relative error")
        plt.legend()
        plt.show()

        return 
        K = 10

        def f(*args):

            # return 1

            return np.linalg.norm(args, ord=1) * np.linalg.norm(args)
        
        def get_range(*args):

            args = np.array(args)
            height_1 = (0 - np.sum(args ** 2)) ** .5
            height_K = (K - np.sum(args ** 2)) ** .5

            return height_1, height_K

        N = 2
        result = 2 * scipy.integrate.nquad(f, [get_range] * (N - 1) + [(0, K)])
        print(result)