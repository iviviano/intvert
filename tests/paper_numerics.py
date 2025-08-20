import numpy as np
import gmpy2 as mp
import sympy as sp
import unittest

from context import binvert

import tracemalloc

def get_all_directions(N, M):

    all_directions = {(1, 0): list(zip(range(1, M + 1), [0] * (M)))}

    for direction in sorted(product(range(N), range(1, N // 2 + 1))):

        for known_direction in all_directions:
            if blur.equiv(direction, known_direction, N):
                if len(all_directions[known_direction]) < M:
                    all_directions[known_direction].append(direction)
                break
        else:
            # all_directions.update({direction1: set([direction1])})
            all_directions.update({direction: [direction]})

    for value in all_directions.values():
        value.insert(0, (0, 0))

    return all_directions

def all_mask(N, M):
    mask = np.zeros((N, N), dtype=bool)
    for _, directions in get_all_directions(N, M).items():
        for k, l in directions:
            mask[k, l] = 1
            mask[-k, -l] = 1
    return mask

def find_sums_from_directions(matrix, transform = None, L = None, beta0 = .1, beta1 = None, beta2 = None, beta3 = None, delta = .9972, all_directions = None):

    size = matrix.shape[0]
    transform = blur.mp_dft2(matrix) if transform is None else transform

    # Default values
    beta1 = beta1 if beta1 else size
    beta2_given = beta2 is not None
    beta3_given = beta3 is not None

    all_directions = all_directions if all_directions else blur.get_all_directions(size, L)
    # print("all directions, ", all_directions)

    sums = {}
    # tol = 1e-18
    tol = 10 ** (5 - np.log10(2) * mp.get_context().precision)
    # print("LLL tolerance:", tol)

    sums_guesses = get_dsums(matrix, all_directions)
    # print()
    # print(sums_guesses)
    for direction, directions in all_directions.items():

        M = len(directions) - 1

        beta2 = beta2 if beta2_given else blur.find_beta2(size, M, beta0)
        beta3 = beta3 if beta3_given else 10 ** min(np.floor(-np.log10(beta2)) - np.log10(tol), 2)
        # print(np.floor(np.log10(beta2)), beta3)
        beta0 = max(beta0, 1 / beta3)

        # print(f"LLL SUM RECOVERY FOR DIRECTION: {direction}\n")
        sums_guess = sums_guesses[direction] 

        basis = directional_sums.get_sum_basis_matrix(size, directions, [transform[direction] for direction in directions], sums_guess, ) # try to remove list comprehension here

        # check if guess is correct
        def check_sums(sums): # if beta0 is added, could use that as another condition
            return max(np.abs(basis[size:, :size] @ blur.mp_round(sums) + basis[size:, size])[:-1]) < tol
        
        if check_sums(sums_guess):
            # print("SUM guess was correct!")
            error_guesses = [np.array([0] * size + [beta0])]
        else: # pass truncated basis to arbitrary precision integers to LLL
            try:
                error_guesses = directional_sums.approximate_svp(np.vectorize(int, otypes = [object])(beta3 * basis.transpose() * np.array([1] * size + [beta1] + [beta2] * (len(basis) - (size + 2)) + [beta0])), delta) / beta3 
            except directional_sums.lll.util.ReductionError as e:
                print("LLL failed for direction", direction, "with error", e)
                continue

        for error_guess in error_guesses:
            if np.isclose(abs(error_guess[-1]), beta0, atol=beta0/10):
                break
        else:
            # print("(alpha_N) sum recovery failed for", direction)
            continue

        # for possible_sums in [sums_guess + error_guess, sums_guess - error_guess]:
        for possible_sums in [sums_guess + error_guess[:size], sums_guess - error_guess[:size]]:
            # print("Reconstructed sums candidate: ", possible_sums)
            if check_sums(possible_sums):
                # print(possible_sums)
                # sums.update({direction: np.rint(possible_sums).astype(np.int64)})
                sums.update({direction: blur.mp_round(possible_sums).astype(np.int64)})
                break
        else:
            pass
            # print("(tol) sum recovery failed for", direction)
            # sums.update({direction: None}) # think about how to do this/ whether it should be done

    # print(f"{len(sums)}/{len(all_directions)} directions successfully recovered")
    return sums

def get_dsums(matrix, directions):

    size = matrix.shape[0]

    dsum_matrix = get_dsum_matrix(size, directions)
    flat_matrix = matrix.flatten()

    return {direction: dsum_matrix @ flat_matrix for direction, dsum_matrix in dsum_matrix.items()}

def get_dsum_matrix(size, directions): 

    rows = {}
    for i, j in directions:
        rows.update({(i, j): np.zeros((size, size), dtype = np.int8)})
        rows[(i, j)][j * np.arange(size) % size, - (i * np.arange(size) % size)] = 1

    return {(i, j): np.reshape([np.roll(row, n, axis = bool(j)) for n in range(size)], (size, size ** 2)) for (i, j), row in rows.items()}


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
                current = sp.stats.mstats.gmean([best_upper_bound, current])
            else:
                current = step * current
        else:
            best_upper_bound = current
            current = sp.stats.mstats.gmean([best_lower_bound, current])
    return current, key(current)

def save_fig(fname):
    plt.savefig(f"../figures/{fname}.png", bbox_inches='tight')
    
def blur_all(matrices, M):
    N = matrices.shape[-1]
    dft = blur.mp_dft2(matrices)
    mask = all_mask(N, M)
    dft[:, ~mask] = 0
    idft = blur.mp_idft2(dft)
    blurred_matrices = blur.mp_real(idft)
    
    return blurred_matrices, dft

def solve_and_count_sums(true, L, time_limit = 5, **lll_params):

    blurred = blurring.blur(true, L)

    stdout = sys.stdout
    with open("/dev/null", "w") as f:
        sys.stdout = f
        lll_results = directional_sums.find_sums(blurred, L, **lll_params)
        actual_sums = directional_sums.get_dsums(true, L)

        n_dir = 0
        for direction, sum in lll_results.items():
            n_dir += np.all(actual_sums[direction] == sum)
        
        A, b = directional_sums.get_sum_system(blurred, L, lll_results=lll_results)

        N = blurred.shape[0]
        x = cp.Variable(blurred.shape, boolean = True)

        constraints = [A @ np.reshape(x, N ** 2) == b]
        prob = cp.Problem(objective = cp.Minimize(0), constraints = constraints)

        prob.solve(solver = "GUROBI", **{"TimeLimit": time_limit})
    
    sys.stdout = stdout

    return prob.status == "optimal" and np.allclose(x.value, true), n_dir
        

class Prime(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(98708743)

        self.n_mat = 20
        self.Ns = [19, 31, 47]

        self.matrices = {N: self.rand.integers(0, 2, (self.n_mat, N, N)) for N in self.Ns}

    def test_K_hist(self, N=47):

        matrices = self.matrices[N]
        Ms = np.arange(1, int(np.sqrt(N)) + 1)
        fig, ax = plt.subplots(1, len(Ms), figsize=(12, 5))
        low = N
        high = 0
        for M in Ms:

            blurred_matrices, _ = blur_all(matrices, M)
            all_directions = get_all_directions(N, M)

            Ks = []
            for n in range(self.n_mat):
                matrix = matrices[n]
                blurred = blurred_matrices[n]

                K = {direction: np.linalg.norm(sum) for direction, sum in get_dsums(matrix - blurred.astype(float), all_directions).items()}
                Ks += list(K.values())

            ax[M - 1].hist(Ks, bins='fd', label=f"M={M}", density=True)

            ax[M - 1].vlines(.5 * np.sqrt((N - 2 * M) * N), 0, .25, color='red', label="Theoretical $K$", linestyle='dashed')
            ax[M - 1].vlines(np.mean(Ks), 0, .25, color='yellow', label="Average", linestyle='dashed')

            low = min(low, min(Ks))
            high = max(high, max(Ks))

        for ax in ax.flatten():
            ax.set_xlabel(r"$K$")
            ax.set_xlim(low, high)
            ax.set_ylim(0, .25)
            # ax.legend()

        plt.subplots_adjust(wspace=0.6)

        save_fig("K_hist_n=%d" % self.n_mat)
        plt.show()

        
    def test_beta_hist(self, n_betas = 10, N = 47, M=None):

        results = pd.read_pickle(f"../data/beta_hist_prime_nmat={self.n_mat}_nbeta={n_betas}")

        with mp.get_context() as context:

            context.precision = 400

            M = M if M else range(1, int(np.sqrt(N)) + 1)

            with self.subTest(N=N):

                matrices = self.matrices[N]

                for M in M:

                    blurred_matrices, dft = blur_all(matrices, M)
                    all_directions = get_all_directions(N, M)

                    with self.subTest(M=M):

                        mag_guess = int(np.log10(directional_sums.find_beta2(N, M, .1)))
                            
                        # beta2s = np.logspace(mag_guess - 3, mag_guess + 3, n_betas)
                        beta2s = np.logspace(mag_guess, mag_guess + 6, n_betas)
                        
                        n_correct = []
                        n_wrong = []
                        n_missed = []
                        for n_beta in range(n_betas):

                            beta2 = beta2s[n_beta]

                            n_correct.append(0)
                            n_wrong.append(0)
                            n_missed.append(0)

                            for n in range(self.n_mat):
                                    
                                matrix = matrices[n]
                                blurred = blurred_matrices[n]
                                lll_results = find_sums_from_directions(blurred, transform=dft[n], beta2=beta2, all_directions=all_directions)
                                # lll_results = find_sums_from_directions(matrix.astype(float), transform=dft[n], beta2=beta2, all_directions=all_directions)

                                actual_sums = get_dsums(matrix, all_directions)

                                for direction in all_directions:
                                    if direction in lll_results:
                                        if np.all(lll_results[direction] == actual_sums[direction]):
                                            n_correct[-1] += 1
                                        else:
                                            print("WRONG", direction, lll_results[direction], actual_sums[direction])
                                            n_wrong[-1] += 1
                                    else:
                                        n_missed[-1] += 1	

                            results[(N, M)].update({
                                "beta": beta2s,
                                "correct": n_correct,
                                "wrong": n_wrong,
                                "missed": n_missed
                            })
                            # results.to_pickle(f"../data/beta_hist_prime_nmat={self.n_mat}_nbeta={n_betas}")

                        dft, blurred = None, None

                        print(f"N = {N}, M = {M}, correct: {n_correct}, missed: {n_missed}")


                        

    
    def test_make_beta_hist(self, n_betas=10, N=47):

        fname = f"../data/beta_hist_prime_nmat={self.n_mat}_nbeta={n_betas}"

        try:
            open(fname)

        except FileNotFoundError:

            results = {(N, M): {"beta": [0] * n_betas, "correct": [0] * n_betas, "wrong": [0] * n_betas, "missed": [0] * n_betas} for N in self.Ns for M in range(1, int(np.sqrt(N)) + 1)}

            results = pd.DataFrame(results)

            results.to_pickle(fname)

        finally:

            results = pd.read_pickle(fname)

        print(results[N])

        Ms = range(1, int(np.sqrt(N)) + 1)

        fig, ax = plt.subplots(1, len(Ms), figsize=(12, 5))

        for M in Ms: 
            data = results[(N, M)]

            ax[M - 1].plot(data['beta'], np.array(data['correct']) / (self.n_mat * (N + 1)))

            ax[M - 1].set_xscale("log")
            ax[M - 1].set_xlabel(r"$\beta_2$")
            
            ax[M - 1].set_title(r"$M = %d$" % M)

            theoretical_beta2 = directional_sums.find_beta2(N, M, .1)
            ax[M - 1].axvline(theoretical_beta2, color='red', linestyle='dashed', label="Theoretical $\\beta_2$")
        
        plt.suptitle(r"$N = %d$" % N)
        plt.subplots_adjust(wspace=0.6)
        save_fig("beta_hist_prime_n=%d" % self.n_mat)
        plt.show()

    def test_percentiles(self, step = 1e5, n_steps = 30, N = 47, M = [5], targets = None):

        # tracemalloc.start()
        # snapshots = [tracemalloc.take_snapshot()]
        with mp.get_context() as context:

            results = pd.read_pickle(f"../data/percentiles_prime_n={self.n_mat}")

            context.precision = 100

            matrices = self.matrices[N]

            for M in M if M else range(1, int(np.sqrt(N)) + 1):
            # for M in range(1, 3):

                blurred_matrices, dft = blur_all(matrices, M)
                all_directions = get_all_directions(N, M)

                with self.subTest(N=N, M=M):

                    def key(beta2):
                        correct_count = 0
                        for n in range(self.n_mat):
                            matrix = matrices[n]
                            blurred = blurred_matrices[n]
                            lll_results = find_sums_from_directions(blurred, dft[n], beta2=beta2, all_directions=all_directions)
                            for direction, sums in get_dsums(matrix, all_directions).items():
                                if direction in lll_results:
                                    if np.all(lll_results[direction] == sums):
                                        correct_count += 1
                        return correct_count / (self.n_mat * len(all_directions))

                    percentiles = []
                    targets = targets if targets else [50, 90, 100]
                    for target in targets:

                        percentile, value = exp_search(step, target / 100, key, n_steps)
                        percentiles.append(percentile)

                        print(f"{N}x{N}, M={M}, {target}th percentile: {percentile:.2e} (value: {value:.2f})")

                        results[(N, M)].update({target: percentile})
                        results.to_pickle(f"../data/percentiles_prime_n={self.n_mat}")

                    blurred_matrices = None
                    dft = None

                    # results[(N, M)].update(dict(zip(targets, percentiles)))
                    # results.to_pickle(f"../data/percentiles_prime_n={self.n_mat}")
                    
                    gc.collect()
                    # snapshots.append(tracemalloc.take_snapshot())
                    # num = 3
                    # for stat in ["filename", "lineno", "traceback"]:
                    #     print(f"[ Top {num} differences in {stat} ]")
                    #     for stat in snapshots[-1].compare_to(snapshots[-2], stat)[:num]:
                    #     # for stat in snapshots[-1].compare_to(snapshots[0], stat)[:num]:
                    #         print(stat)

            
        # tracemalloc.stop()

    def make_percentile_table(self):

        fname = f"../data/percentiles_prime_n={self.n_mat}"

        try:

            open(fname)

        except FileNotFoundError:

            table = {(N, M): {
                "Theory": directional_sums.find_beta2(N, M, .1),
                50: 0,
                90: 0,
                100: 0
                } for N in self.Ns for M in range(1, int(np.sqrt(N)) + 1)}

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

        pipe = '\\textbar'
        for N in self.Ns:

            print("\midrule")

            for M in range(1, int(np.sqrt(N)) + 1):

                row = (pipe if M - 1 else f"${N}$") + f" & ${M}$ & {table[(N, M)].loc['Theory']:.2e}"

                for target in targets:
                    row += f" & {table[(N, M)].loc[target]:.2e}"

                print(row + " \\\\")

        print("\\end{tabular}")
    
    def make_percentile_plot(self):

        table = pd.read_pickle(f"../data/percentiles_prime_n={self.n_mat}")

        # ax = table.plot(x="M", y=["Theory", 50, 90, 100], logy=True, subplots=True, layout=(1, len(self.Ns)), marker='o', title="Percentiles of $\\beta_2$ for prime $N$")
        fig, ax = plt.subplots(1, len(self.Ns), figsize=(12, 5))

        for i, N in enumerate(self.Ns):
            ax[i].set_title(r"$N = %d$" % N)
            print(table[N])
            ax[i].plot(table[N].T, label=["Theory", "50th", "90th", "100th"], marker='o')

            ax[i].set_yscale("log")
            ax[i].set_xlabel(r"$M$")
            ax[i].set_ylabel(r"$\beta_2$")
            ax[i].legend()

            ax[i].set_xticks(range(1, int(np.sqrt(N)) + 1))

        plt.subplots_adjust(wspace=0.6)

        save_fig("percentiles_prime_n=%d" % self.n_mat)
        plt.show()


    def test_K(self, N=47, M=range(6, 7)):

        with mp.get_context() as context:

            result = pd.read_pickle(f"../data/K_prime_n={self.n_mat}")

            context.precision = 100

            L = int(np.sqrt(N))

            matrices = self.rand.integers(0, 2, (self.n_mat, N, N))

            n_correct = dict(zip(range(1, L + 1), [0] * L))
            n_wrong = dict(zip(range(1, L + 1), [0] * L))
            n_missed = dict(zip(range(1, L + 1), [0] * L))
            
            for M in M if M else range(1, L + 1):
                # blurred = blur.blur(matrices, l)

                    dft = blur.mp_dft2(matrices)
                    mask = all_mask(N, M)
                    dft[:, ~mask] = 0
                    idft = blur.mp_idft2(dft)
                    blurred_matrices = blur.mp_real(idft)
                    idft = None
                    all_directions = get_all_directions(N, M)

                    for n in range(self.n_mat):

                        matrix = matrices[n]
                        blurred = blurred_matrices[n]

                        K = {direction: np.linalg.norm(sum) for direction, sum in get_dsums(matrix - blurred.astype(float), all_directions).items()}

                        for direction, sums in get_dsums(matrix, all_directions).items():
                            beta2 = directional_sums.find_beta2(N, M, .1, K[direction])
                            lll_results = find_sums_from_directions(blurred, dft[n], all_directions={direction: all_directions[direction]}, beta2=beta2)
                            if direction in lll_results:
                                if np.all(lll_results[direction] == sums):
                                    n_correct[M] += 1
                                else:
                                    n_wrong[M] += 1
                            else:
                                n_missed[M] += 1
                    
                    print(f"N={N}, M={M}, correct: {n_correct[M]}, wrong: {n_wrong[M]}, missed: {n_missed[M]}")

                    result.loc[result['N'] == N, M] = n_correct[M]
                    result.to_pickle(f"../data/K_prime_n={self.n_mat}")



    def make_K_table(self):

        fname = f"../data/K_prime_n={self.n_mat}"

        try:

            open(fname)

        except FileNotFoundError:

            table = {"N": self.Ns} | {M: [np.nan] * len(self.Ns) for M in range(1, int(np.sqrt(max(self.Ns))) + 1)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname)

        print(table)

        print("\\begin{tabular}{c|%s}" % ("c" * (len(table.columns) - 1)))

        header = "\diagbox{$N$}{$M$}"
        for M in range(1, int(np.sqrt(max(self.Ns))) + 1):
            header += f" & {M}"
        print(header + " \\\\")

        print("\\midrule")

        # pipe = '\\textbar'

        for N in self.Ns:
            row = f"${N}$"
            for M in range(1, int(np.sqrt(max(self.Ns))) + 1):
                value = table.loc[table['N'] == N, M].values[0]
                row += f" & {'|' if np.isnan(value) else value / (self.n_mat * (N + 1)) * 100:.4}"
            print(row + " \\\\")

        print("\\end{tabular}")

    def test_full(self, N = 47):

        L = int(N ** .5) + 4

        fname = f"../data/full_prime_n={self.n_mat}"
        results = pd.read_pickle(fname)

        all = False
        matrices = self.rand.integers(0, 2, (self.n_mat, N, N))
        while not all:
            print(f"Testing L = {L}")

            n_rec = 0
            n_dir = 0
            ttime = 0
 
            stdout = sys.stdout
            for n in range(self.n_mat):

                # with open("log", "w") as file:
                #     sys.stdout = file
                #     stime = time()
                #     deblurred = deblur_binary(blurred[n], L, solver='GUROBI', beta2=1e12, beta3=1e2)
                #     ttime += time() - stime

                # sys.stdout = stdout

                # with open("log", "r") as file:
                #     while not (lll_output := re.match(r"\d*/\d* directions successfully recovered\s*", file.readline())): pass

                # n_dir += int(re.search(r"\d*", lll_output.group()).group())

                # if np.all(deblurred == matrices[n]):
                #     n_rec += 1

                stime = time()
                correct, n_corr_dir = solve_and_count_sums(matrices[n], L, beta2=1e12, beta3=1e2)
                ttime += time() - stime
                n_rec += correct
                n_dir += n_corr_dir

            results[(N, L)].update({"n_rec": n_rec, "n_dir": n_dir, "time": ttime})
            results.to_pickle(fname)
            print(results)
            
            all = n_rec == self.n_mat
            L += 1

        # os.remove("log")


    def make_full_table(self):

        fname = f"../data/full_prime_n={self.n_mat}"

        try:

            open(fname)

        except FileNotFoundError:

            table = {(N, L): {
                "n_rec": np.nan,
                "time": np.nan,
                "n_dir": np.nan
                } for N in self.Ns for L in range(int(np.sqrt(N)), N // 2)}

            table = pd.DataFrame(table)

            table.to_pickle(fname)

        finally:

            table = pd.read_pickle(fname) 

        print(table)
        
        for N in self.Ns:
            print(table[N])
            print(table[(N, int(N ** .5))])
        results = table.dropna(axis=1)
        print(results)

        print("\\begin{tabular}{cc|%s}" % ("c" * (len(table[(19, 5)].index))))

        print("N & L & Rec, \% & $t$, sec & Dir, \# \\\\")

        print("\midrule\midrule")

        for N in self.Ns:
            
            for L in range(int(N ** .5), N // 2):
                # print(table[(N, L)])
                # print(results[(N, L)])
                try:
                    print(f"{N} & {L} & {results[(N, L)]['n_rec']} & {results[(N, L)]['time'] / self.n_mat:.1f} & {results[(N, L)]['n_dir'] / self.n_mat:.1f} \\\\")
                except KeyError as err:
                    # print(err)
                    break

            print("\midrule")

        print("\\end{tabular}")


    def test_full_precision(self, N = 47):

        fname = f"../data/full_prec_n={self.n_mat}"

        results = pd.read_pickle(fname)

        matrices = self.rand.integers(0, 2, (self.n_mat, N, N))
        for prec in results[N].columns[2:]:

            L = int(N ** .5)

            Ls = []
            n_recs = []
            n_dirs = []
            with mp.get_context() as context:
                if prec:
                    context.precision = int(prec * np.log2(10))
                print(f"Precision: {prec} decimal digits ({context.precision} bits)")

                all = False
                while not all:
                    print(f"Testing L = {L}")

                    n_rec = 0
                    n_dir = 0
        
                    stdout = sys.stdout
                    beta2 = 10 ** (context.precision / np.log2(10) - 4)
                    print(f"{beta2:.2}")
                    for n in range(self.n_mat):
                        correct, n_corr_dir = solve_and_count_sums(matrices[n], L, 15, beta2=beta2, beta3=1e2) # should time limit depend on N?
                        n_rec += correct
                        n_dir += n_corr_dir
    
                        # try:
                        #     correct, n_corr_dir = solve_and_count_sums(matrices[n], L, 5, beta2=1e12, beta3=1e2) # should time limit depend on N?
                        #     n_rec += correct
                        #     n_dir += n_corr_dir
                        # except ValueError as err:
                        #     print(err, file=stdout)
                        #     pass
                        # finally:
                        #     sys.stdout = stdout

                    Ls.append(L)
                    n_recs.append(n_rec)
                    n_dirs.append(n_dir)
                    results[(N, prec)].update({"L": Ls, "n_rec": n_recs, "n_dir": n_dirs})
                    results.to_pickle(fname)
                    
                    all = n_rec == self.n_mat
                    L += 1                
                print(results)


            
    def make_full_precision_plot(self, N = 19):

        precs = [0, 5, 10] # number of decimal digits

        fname = f"../data/full_prec_n={self.n_mat}"

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
                data = np.concatenate([results[var] / self.n_mat / (1 + N * i), np.ones(len(Ls) - len(results["L"]))])
                ax[i].plot(Ls, data, label=f"prec={prec}")
            ax[i].legend()

            ax[i].set_title(["Percentage of test matrices recovered", "Percentage of directions recovered"][i])

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

    def draw_subgroup_lattice2D(self, M=30, N=1, cyclic=True):

        assert cyclic

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
                print(f"{n}/{stretch * (2 * i - (len(ns) - 1))}/{-layer}/{groups[n]},")

            divisors = new_divisors
            layer += 1

        print(r"} { \node (\n) at (\x,\y) {\set\group}; }")

        print(r"  \foreach \a/\b in {")
        for d1, d2 in connections:
            print(f"{d1}/{d2},")
        print(r"  } { \draw (\a) -- (\b); }")
        
        print(r"\end{tikzpicture}")
