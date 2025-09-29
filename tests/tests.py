from context import intvert

import numpy as np
import gmpy2 as mp
import unittest
from itertools import product

class TestDFT(unittest.TestCase):

    def test_dft(self):

        rand = np.random.default_rng(3242243)

        mp_dft, mp_idft, mp_dft2, mp_idft2 = intvert.mp_dft, intvert.mp_idft, intvert.mp_dft2, intvert.mp_idft2, 

        signal = np.array([5])
        self.assertTrue(np.allclose(signal - mp_dft(signal), 0), "1D length 1 dft")
        self.assertTrue(np.allclose(signal - mp_idft(signal), 0), "1D length 1 dft")
        
        signal = np.array([[5]])
        self.assertTrue(np.allclose(signal - mp_dft2(signal), 0), "2D 1x1 dft")
        self.assertTrue(np.allclose(signal - mp_idft2(signal), 0), "2D 1x1 idft")

        with mp.get_context() as c:
            c.precision = 100
            for M, N in product(range(1, 15), range(1, 25, 3)):

                matrices = rand.random((5, M, N))

                fft = np.fft.fft(matrices)
                self.assertTrue(np.allclose(fft - mp_dft(matrices), 0), "dft close to np")

                fft2 = np.fft.fft2(matrices)
                self.assertTrue(np.allclose(fft2 - mp_dft2(matrices), 0), "dft2 close to np")

                ifft = np.fft.ifft(matrices)
                self.assertTrue(np.allclose(ifft - mp_idft(matrices), 0), "idft close to np")

                ifft2 = np.fft.ifft2(matrices)
                self.assertTrue(np.allclose(ifft2 - mp_idft2(matrices), 0), "idft2 close to np")

                
            self.assertLess(np.sum(abs(matrices - mp_idft2(mp_dft2(matrices)))), np.sum(abs(matrices - np.fft.ifft2(np.fft.ifft2(matrices)))) / 10, "DFT and inverse DFT introduces less error than numpy")


class TestCoeffs(unittest.TestCase): # Good

    def test_get_coeff_classes_1D(self):

        get_coeff_classes_1D = intvert.get_coeff_classes_1D

        expected = {
            1: {1: {0}},
            2: {2: {0}, 1: {1}},
            3: {3: {0}, 1: {1, 2}},
            4: {4: {0}, 1: {1, 3}, 2: {2}},
            5: {5: {0}, 1: {1, 2, 3, 4}},
            30: {
                30: {0}, 
                1: {1, 7, 11, 13, 17, 19, 23, 29}, 
                2: {2, 4, 8, 14, 16, 22, 26, 28}, 
                3: {3, 9, 21, 27},
                5: {5, 25},
                6: {6, 12, 18, 24},
                10: {10, 20},
                15: {15}
                }
        }

        for N in expected:
            with self.subTest(N=N, conjugates=True):
                actual = get_coeff_classes_1D(N, True)
                self.assertDictEqual(actual, expected[N])


        expected = {
            1: {1: {0}},
            2: {2: {0}, 1: {1}},
            3: {3: {0}, 1: {1}},
            4: {4: {0}, 1: {1}, 2: {2}},
            5: {5: {0}, 1: {1, 2}},
            30: {
                30: {0}, 
                1: {1, 7, 11, 13},
                2: {2, 4, 8, 14},
                3: {3, 9},
                5: {5},
                6: {6, 12},
                10: {10},
                15: {15}
                }
        }

        for N in expected:
            with self.subTest(N=N, conjugates=False):
                actual = get_coeff_classes_1D(N, False)
                self.assertDictEqual(actual, expected[N])
                

    def test_get_coeff_classes_2D(self): 

        get_coeff_classes_2D = intvert.get_coeff_classes_2D

        expected = {
            (1, 1): {(1, 1): {frozenset({(0, 0)})}},
            (1, 2): {
                (1, 2): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(0, 1)})},
                },
            (2, 1): {
                (2, 1): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 0)})},
                },
            (3, 3): {
                (3, 3): {frozenset({(0, 0)})},
                 (1, 3): {frozenset([(1, 0), (2, 0)])},
                 (3, 1): {frozenset([(0, 1), (0, 2)])},
                (1, 1): {frozenset([(1, 1), (2, 2)]), frozenset([(1, 2), (2, 1)])}
                 },
            (3, 2): {
                (3, 2): {frozenset({(0, 0)})},
                (1, 2): {frozenset({(1, 0), (2, 0)})},
                (3, 1): {frozenset({(0, 1)})},
                (1, 1): {frozenset({(1, 1), (2, 1)})},
            },
            (5, 4): {
                (5, 4): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1), (2, 3), (3, 3), (4, 1), (1, 3), (3, 1), (2, 1), (4, 3)})},
                (1, 2): {frozenset({(1, 2), (3, 2), (2, 2), (4, 2)})},
                (1, 4): {frozenset({(1, 0), (2, 0), (3, 0), (4, 0)})},
                (5, 1): {frozenset({(0, 1), (0, 3)})},
                (5, 2): {frozenset({(0, 2)})},
            },
            (6, 6): {
                (6, 6): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1), (5, 5)}), frozenset({(1, 5), (5, 1)})},
                (1, 2): {frozenset({(1, 2), (5, 4)}), frozenset({(1, 4), (5, 2)})},
                (2, 1): {frozenset({(2, 1), (4, 5)}), frozenset({(4, 1), (2, 5)})},
                (2, 2): {frozenset({(2, 2), (4, 4)}), frozenset({(2, 4), (4, 2)})},
                (1, 3): {frozenset({(1, 3), (5, 3)})},
                (3, 1): {frozenset({(3, 1), (3, 5)})},
                (2, 3): {frozenset({(2, 3), (4, 3)})},
                (3, 2): {frozenset({(3, 2), (3, 4)})},
                (3, 3): {frozenset({(3, 3)})},
                (1, 6): {frozenset({(1, 0), (5, 0)})},
                (6, 1): {frozenset({(0, 1), (0, 5)})},
                (2, 6): {frozenset({(2, 0), (4, 0)})},
                (6, 2): {frozenset({(0, 2), (0, 4)})},
                (3, 6): {frozenset({(3, 0)})},
                (6, 3): {frozenset({(0, 3)})},
            },
            (4, 6): {
                (4, 6): {frozenset({(0, 0)})},
                (2, 6): {frozenset({(2, 0)})},
                (1, 6): {frozenset({(1, 0), (3, 0)})},
                (4, 3): {frozenset({(0, 3)})},
                (4, 2): {frozenset({(0, 2), (0, 4)})},
                (4, 1): {frozenset({(0, 1), (0, 5)})},
                (2, 3): {frozenset({(2, 3)})},
                (2, 2): {frozenset({(2, 2), (2, 4)})},
                (2, 1): {frozenset({(2, 1), (2, 5)})},
                (1, 3): {frozenset({(1, 3), (3, 3)})},
                (1, 2): {frozenset({(1, 2), (3, 4), (3, 2), (1, 4)})},
                (1, 1): {frozenset({(1, 1), (1, 5), (3, 1), (3, 5)})},
            },
        }

        for M, N in expected:
            with self.subTest(M=M, N=N, conjugates=True):
                actual = get_coeff_classes_2D(M, N, include_conjugates=True)
                self.assertDictEqual(actual, expected[M, N])

        
        expected = {
            (1, 1): {(1, 1): {frozenset({(0, 0)})}},
            (1, 2): {
                (1, 2): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(0, 1)})},
                },
            (2, 1): {
                (2, 1): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 0)})},
                },
            (3, 3): {
                (3, 3): {frozenset({(0, 0)})},
                 (1, 3): {frozenset([(1, 0)])},
                 (3, 1): {frozenset([(0, 1)])},
                (1, 1): {frozenset([(1, 1)]), frozenset([(1, 2)])}
                 },
            (3, 2): {
                (3, 2): {frozenset({(0, 0)})},
                (1, 2): {frozenset({(1, 0)})},
                (3, 1): {frozenset({(0, 1)})},
                (1, 1): {frozenset({(1, 1)})},
            },
            (5, 4): {
                (5, 4): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1), (2, 3), (2, 1), (1, 3)})},
                (1, 2): {frozenset({(1, 2), (2, 2)})},
                (1, 4): {frozenset({(1, 0), (2, 0)})},
                (5, 1): {frozenset({(0, 1)})},
                (5, 2): {frozenset({(0, 2)})},
            },
            (6, 6): {
                (6, 6): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1)}), frozenset({(1, 5)})},
                (1, 2): {frozenset({(1, 2)}), frozenset({(1, 4)})},
                (2, 1): {frozenset({(2, 1)}), frozenset({(2, 5)})},
                (2, 2): {frozenset({(2, 2)}), frozenset({(2, 4)})},
                (1, 3): {frozenset({(1, 3)})},
                (3, 1): {frozenset({(3, 1)})},
                (2, 3): {frozenset({(2, 3)})},
                (3, 2): {frozenset({(3, 2)})},
                (3, 3): {frozenset({(3, 3)})},
                (1, 6): {frozenset({(1, 0)})},
                (6, 1): {frozenset({(0, 1)})},
                (2, 6): {frozenset({(2, 0)})},
                (6, 2): {frozenset({(0, 2)})},
                (3, 6): {frozenset({(3, 0)})},
                (6, 3): {frozenset({(0, 3)})},
            },
            (4, 6): {
                (4, 6): {frozenset({(0, 0)})},
                (2, 6): {frozenset({(2, 0)})},
                (1, 6): {frozenset({(1, 0)})},
                (4, 3): {frozenset({(0, 3)})},
                (4, 2): {frozenset({(0, 2)})},
                (4, 1): {frozenset({(0, 1)})},
                (2, 3): {frozenset({(2, 3)})},
                (2, 2): {frozenset({(2, 2)})},
                (2, 1): {frozenset({(2, 1)})},
                (1, 3): {frozenset({(1, 3)})},
                (1, 2): {frozenset({(1, 2), (1, 4)})},
                (1, 1): {frozenset({(1, 1), (1, 5)})},
            },
            }

        for M, N in expected:
            with self.subTest(M=M, N=N, conjugates=False):
                actual = get_coeff_classes_2D(M, N, include_conjugates=False)
                # print(actual)
                self.assertDictEqual(actual, expected[M, N])


    def test_select_coeffs_1D(self):

        select_coeffs = intvert.select_coeffs_1D

        expected = {
            1: {1: {0}},
            2: {2: {0}, 1: {1}},
            3: {3: {0}, 1: {1}},
            4: {4: {0}, 1: {1}, 2: {2}},
            5: {5: {0}, 1: {1}},
            6: {6: {0}, 1: {1}, 2: {2}, 3: {3}},
            30: {30: {0}, 1: {1}, 2: {2}, 3: {3}, 5: {5}, 6: {6}, 10: {10}, 15: {15}}
        }

        for N in expected:
            for L in [[], 1, [1]]:
                with self.subTest(N=N, Ls=L):
                    actual = select_coeffs(N, Ls=L)
                    self.assertDictEqual(actual, expected[N])


        expected = {
            1: {1: {0}},
            5: {5: {0}, 1: {1, 2}},
            6: {6: {0}, 1: {1}, 2: {2}, 3: {3}},
            7: {7: {0}, 1: {1, 2}},
            10: {10: {0}, 1: {1, 3}, 2: {2}, 5: {5}},
            20: {20: {0}, 1: {1, 3}, 2: {2}, 4: {4}, 5: {5}, 10: {10}},
            25: {25: {0}, 1: {1, 2}, 5: {5}},
            30: {30: {0}, 1: {1, 7}, 2: {2}, 3: {3}, 5: {5}, 6: {6}, 10: {10}, 15: {15}},
        }

        for N in expected:
            for L in [[2]]:
                with self.subTest(N=N, L=L):
                    actual = select_coeffs(N, Ls=L)
                    self.assertDictEqual(actual, expected[N])

        
        expected = {
            1: {1: {0}},
            5: {5: {0}, 1: {1, 2}},
            6: {6: {0}, 1: {1}, 2: {2}, 3: {3}},
            7: {7: {0}, 1: {1, 2}},
            10: {10: {0}, 1: {1, 3}, 2: {2, 4}, 5: {5}},
            20: {20: {0}, 1: {1, 3}, 2: {2, 6}, 4: {4, 8}, 5: {5}, 10: {10}},
            25: {25: {0}, 1: {1, 2}, 5: {5, 10}},
            30: {30: {0}, 1: {1, 7}, 2: {2, 4}, 3: {3, 9}, 5: {5}, 6: {6, 12}, 10: {10}, 15: {15}},
        }

        for N in expected:
            for L in [2]:
                with self.subTest(N=N, L=L):
                    actual = select_coeffs(N, L)
                    self.assertDictEqual(actual, expected[N])


        expected = {
            1: {1: {0}},
            5: {5: {0}, 1: {1, 2}},
            6: {6: {0}, 1: {1}, 2: {2}, 3: {3}},
            7: {7: {0}, 1: {1, 2, 3}},
            10: {10: {0}, 1: {1, 3}, 2: {2, 4}, 5: {5}},
            20: {20: {0}, 1: {1, 3, 7}, 2: {2, 6}, 4: {4}, 5: {5}, 10: {10}},
            25: {25: {0}, 1: {1, 2, 3}, 5: {5, 10}},
            30: {30: {0}, 1: {1, 7, 11}, 2: {2, 4}, 3: {3, 9}, 5: {5}, 6: {6}, 10: {10}, 15: {15}},
            35: {35: {0}, 1: {1, 2, 3}, 5: {5, 10}, 7: {7, 14}},
            64: {64: {0}, 1: {1, 3, 5}, 2: {2, 6}, 4: {4}, 8: {8}, 16: {16}, 32: {32}},
        }

        for N in expected:
            L = [3, 2]
            with self.subTest(N=N, L=L):
                actual = select_coeffs(N, Ls=L)
                self.assertDictEqual(actual, expected[N])

        for Ls in [5, [5]]:
            with self.subTest(N=59, Ls=Ls):
                actual = select_coeffs(59, Ls=Ls)
                self.assertDictEqual(actual, {59: {0}, 1: {1, 2, 3, 4, 5}})


    def test_select_coeffs_2D(self): 

        expected = { # Ls = 1
            (1, 1): {(1, 1): {frozenset({(0, 0)})}},
            (3, 3): {
                (3, 3): {frozenset({(0, 0)})},
                 (1, 3): {frozenset([(1, 0)])},
                 (3, 1): {frozenset([(0, 1)])},
                (1, 1): {frozenset([(1, 1)]), frozenset([(1, 2)])}
                 },
            (5, 4): {
                (5, 4): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1)})},
                (1, 2): {frozenset({(1, 2)})},
                (1, 4): {frozenset({(1, 0)})},
                (5, 1): {frozenset({(0, 1)})},
                (5, 2): {frozenset({(0, 2)})},
            },
            (6, 6): {
                (6, 6): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1)}), frozenset({(1, 5)})},
                (1, 2): {frozenset({(1, 2)}), frozenset({(1, 4)})},
                (2, 1): {frozenset({(2, 1)}), frozenset({(2, 5)})},
                (2, 2): {frozenset({(2, 2)}), frozenset({(2, 4)})},
                (1, 3): {frozenset({(1, 3)})},
                (3, 1): {frozenset({(3, 1)})},
                (2, 3): {frozenset({(2, 3)})},
                (3, 2): {frozenset({(3, 2)})},
                (3, 3): {frozenset({(3, 3)})},
                (1, 6): {frozenset({(1, 0)})},
                (6, 1): {frozenset({(0, 1)})},
                (2, 6): {frozenset({(2, 0)})},
                (6, 2): {frozenset({(0, 2)})},
                (3, 6): {frozenset({(3, 0)})},
                (6, 3): {frozenset({(0, 3)})},
            },
            (6, 7): {
                (6, 7): {frozenset({(0, 0)})},
                (6, 1): {frozenset({(0, 1)})},
                (3, 7): {frozenset({(3, 0)})},
                (2, 7): {frozenset({(2, 0)})},
                (1, 7): {frozenset({(1, 0)})},
                (3, 1): {frozenset({(3, 1)})},
                (2, 1): {frozenset({(2, 1)})},
                (1, 1): {frozenset({(1, 1)})},
            },
            (5, 5): {
                (5, 5): {frozenset({(0, 0)})},
                (1, 5): {frozenset({(1, 0)})},
                (5, 1): {frozenset({(0, 1)})},
                (1, 1): {frozenset({(1, 1)}), frozenset({(1, 2)}), frozenset({(1, 3)}), frozenset({(1, 4)})},
            },
            (4, 6): {
                (4, 6): {frozenset({(0, 0)})},
                (2, 6): {frozenset({(2, 0)})},
                (1, 6): {frozenset({(1, 0)})},
                (4, 3): {frozenset({(0, 3)})},
                (4, 2): {frozenset({(0, 2)})},
                (4, 1): {frozenset({(0, 1)})},
                (2, 3): {frozenset({(2, 3)})},
                (2, 2): {frozenset({(2, 2)})},
                (2, 1): {frozenset({(2, 1)})},
                (1, 3): {frozenset({(1, 3)})},
                (1, 2): {frozenset({(1, 2)})},
                (1, 1): {frozenset({(1, 1)})},
            }
        }

        for M, N in expected:
            with self.subTest(M=M, N=N, Ls=1):
                actual = intvert.select_coeffs_2D(M, N)
                self.assertDictEqual(actual, expected[M, N])
                actual = intvert.select_coeffs_2D(M, N, 1)
                self.assertDictEqual(actual, expected[M, N])
                actual = intvert.select_coeffs_2D(M, N, [1])
                self.assertDictEqual(actual, expected[M, N])
                actual = intvert.select_coeffs_2D(M, N, [1, 1])
                self.assertDictEqual(actual, expected[M, N])

        
        expected = { # Ls = 2
            (5, 4): {
                (5, 4): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1), (1, 3)})},
                (1, 2): {frozenset({(1, 2), (2, 2)})},
                (1, 4): {frozenset({(1, 0), (2, 0)})},
                (5, 1): {frozenset({(0, 1)})},
                (5, 2): {frozenset({(0, 2)})},
            },
            (6, 7): {
                (6, 7): {frozenset({(0, 0)})},
                (6, 1): {frozenset({(0, 1), (0, 2)})},
                (3, 7): {frozenset({(3, 0)})},
                (2, 7): {frozenset({(2, 0)})},
                (1, 7): {frozenset({(1, 0)})},
                (3, 1): {frozenset({(3, 1), (3, 2)})},
                (2, 1): {frozenset({(2, 1), (2, 2)})},
                (1, 1): {frozenset({(1, 1), (1, 2)})},
            },
            (5, 5): {
                (5, 5): {frozenset({(0, 0)})},
                (1, 5): {frozenset({(1, 0), (2, 0)})},
                (5, 1): {frozenset({(0, 1), (0, 2)})},
                (1, 1): {frozenset({(1, 1), (2, 2)}), frozenset({(1, 2), (2, 4)}), frozenset({(1, 3), (2, 1)}), frozenset({(1, 4), (2, 3)})},
            },
            (4, 6): {
                (4, 6): {frozenset({(0, 0)})},
                (2, 6): {frozenset({(2, 0)})},
                (1, 6): {frozenset({(1, 0)})},
                (4, 3): {frozenset({(0, 3)})},
                (4, 2): {frozenset({(0, 2)})},
                (4, 1): {frozenset({(0, 1)})},
                (2, 3): {frozenset({(2, 3)})},
                (2, 2): {frozenset({(2, 2)})},
                (2, 1): {frozenset({(2, 1)})},
                (1, 3): {frozenset({(1, 3)})},
                (1, 2): {frozenset({(1, 2), (1, 4)})},
                (1, 1): {frozenset({(1, 1), (1, 5)})},
            }
        }

        for M, N in expected:
            with self.subTest(M=M, N=N, Ls=2):
                actual = intvert.select_coeffs_2D(M, N, 2)
                self.assertDictEqual(actual, expected[M, N])


        expected = { # Ls = [2]
            (5, 4): {
                (5, 4): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1), (1, 3)})},
                (1, 2): {frozenset({(1, 2)})},
                (1, 4): {frozenset({(1, 0)})},
                (5, 1): {frozenset({(0, 1)})},
                (5, 2): {frozenset({(0, 2)})},
            },
            (6, 7): {
                (6, 7): {frozenset({(0, 0)})},
                (6, 1): {frozenset({(0, 1)})},
                (3, 7): {frozenset({(3, 0)})},
                (2, 7): {frozenset({(2, 0)})},
                (1, 7): {frozenset({(1, 0)})},
                (3, 1): {frozenset({(3, 1)})},
                (2, 1): {frozenset({(2, 1)})},
                (1, 1): {frozenset({(1, 1), (1, 2)})},
            },
            (5, 5): {
                (5, 5): {frozenset({(0, 0)})},
                (1, 5): {frozenset({(1, 0), (2, 0)})},
                (5, 1): {frozenset({(0, 1), (0, 2)})},
                (1, 1): {frozenset({(1, 1), (2, 2)}), frozenset({(1, 2), (2, 4)}), frozenset({(1, 3), (2, 1)}), frozenset({(1, 4), (2, 3)})},
            },
            (4, 6): {
                (4, 6): {frozenset({(0, 0)})},
                (2, 6): {frozenset({(2, 0)})},
                (1, 6): {frozenset({(1, 0)})},
                (4, 3): {frozenset({(0, 3)})},
                (4, 2): {frozenset({(0, 2)})},
                (4, 1): {frozenset({(0, 1)})},
                (2, 3): {frozenset({(2, 3)})},
                (2, 2): {frozenset({(2, 2)})},
                (2, 1): {frozenset({(2, 1)})},
                (1, 3): {frozenset({(1, 3)})},
                (1, 2): {frozenset({(1, 2), (1, 4)})},
                (1, 1): {frozenset({(1, 1), (1, 5)})},
            }
        }

        for M, N in expected:
            with self.subTest(M=M, N=N, Ls=[2]):
                actual = intvert.select_coeffs_2D(M, N, [2])
                self.assertDictEqual(actual, expected[M, N])


        expected = { # Ls = [2, 2]
            (5, 4): {
                (5, 4): {frozenset({(0, 0)})},
                (1, 1): {frozenset({(1, 1), (1, 3)})},
                (1, 2): {frozenset({(1, 2), (2, 2)})},
                (1, 4): {frozenset({(1, 0)})},
                (5, 1): {frozenset({(0, 1)})},
                (5, 2): {frozenset({(0, 2)})},
            },
            (6, 7): {
                (6, 7): {frozenset({(0, 0)})},
                (6, 1): {frozenset({(0, 1)})},
                (3, 7): {frozenset({(3, 0)})},
                (2, 7): {frozenset({(2, 0)})},
                (1, 7): {frozenset({(1, 0)})},
                (3, 1): {frozenset({(3, 1), (3, 2)})},
                (2, 1): {frozenset({(2, 1), (2, 2)})},
                (1, 1): {frozenset({(1, 1), (1, 2)})},
            },
        }

        for M, N in expected:
            with self.subTest(M=M, N=N, Ls=[2, 2]):
                actual = intvert.select_coeffs_2D(M, N, [2, 2])
                self.assertDictEqual(actual, expected[M, N])


class Test1DInversion(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(78906)

    def test_small(self):

        for N in range(1, 50):

            with self.subTest(N=N):
                signal = self.rand.integers(0, 2, N)

                blurred = intvert.sample_1D(signal)

                inverted = intvert.invert_1D(blurred)

                self.assertTrue(np.allclose(signal - inverted, 0), f"actual: {inverted}; expected: {signal}")

    def test_large_prime(self):

        for N in [53, 59]:

            with self.subTest(N=N):

                signal = self.rand.integers(0, 2, N)
                blurred = intvert.sample_1D(signal)
                inverted = intvert.invert_1D(blurred)
                self.assertFalse(np.allclose(signal - inverted, 0), f"Too large for 1 coefficient, double precision")

                with mp.get_context() as c:
                    c.precision = 200
                    blurred = intvert.sample_1D(signal)
                    inverted = intvert.invert_1D(blurred, beta2=1e20)
                    self.assertTrue(np.allclose(signal - inverted, 0), f"works with larger beta2; actual: {inverted}; expected: {signal}")

                known_coeffs = intvert.select_coeffs_1D(N, [10])
                blurred = intvert.sample_1D(signal, known_coeffs=known_coeffs)
                inverted = intvert.invert_1D(blurred, known_coeffs=known_coeffs)
                self.assertTrue(np.allclose(signal - inverted, 0), f"Works with several coefficients; actual: {inverted}; expected: {signal}")

    def test_coeffs(self):

        known_coeffs = {
            6: {6: {0}, 1: {5}, 2: {4}, 3: {3}},
            12: {12: {0}, 1: {11, 7}, 2: {10}, 3: {3}, 4: {8}, 6: {6}},
            15: {15: {0}, 1: {14, 13, 11}, 3: {6}, 5: {5}},
            30: {30: {0}, 1: {7, 29}, 2: {8, 14}, 3: {3}, 5: {25}, 6: {12}, 10: {10}, 15: {15}},
        }
        for N in known_coeffs:
            with self.subTest(N=N):
                for _ in range(10):
                    signal = self.rand.integers(0, 2, N)
                    blurred = intvert.sample_1D(signal, known_coeffs=known_coeffs[N])
                    inverted = intvert.invert_1D(blurred, known_coeffs=known_coeffs[N])
                    self.assertTrue(np.allclose(signal - inverted, 0), f"actual: {inverted}; expected: {signal}")
                    inverted = intvert.invert_1D(blurred)
                    self.assertTrue(np.allclose(signal - inverted, 0), f"finds coefficients automatically; actual: {inverted}; expected: {signal}") 

    def test_non_binary(self):

        N = 19
        signal = self.rand.binomial(19, .5, N)
        blurred = intvert.sample_1D(signal)
        inverted = intvert.invert_1D(blurred)
        self.assertTrue(np.allclose(signal - inverted, 0))

    def test_large_M(self):

        M = 7
        for prec in range(53, 100, 50):
            with mp.get_context() as c:
                # c.precision = prec
                for N in range(80, 90):
                    known_coeffs = intvert.select_coeffs_1D(N, M)
                    signal = self.rand.binomial(N, .5, (10, N))
                    with self.subTest(N=N):
                        blurred = intvert.sample_1D(signal, known_coeffs)
                        inverted = intvert.invert_1D(blurred, known_coeffs)
                        self.assertTrue(np.allclose(signal - inverted, 0))


    def test_vectorize(self):

        N = 60

        signals = self.rand.integers(0, 2, (10, 5, N))
        blurred = intvert.sample_1D(signals)
        inverted = intvert.invert_1D(blurred)
        self.assertTrue(np.allclose(signals - inverted, 0))

        known_coeffs = intvert.select_coeffs_1D(N, [2])
        blurred = intvert.sample_1D(signals, known_coeffs)
        inverted = intvert.invert_1D(blurred, known_coeffs=known_coeffs, beta3=1e3)
        self.assertTrue(np.allclose(signals - inverted, 0))


class Test2DInversion(unittest.TestCase):

    def setUp(self):
        
        self.rand = np.random.default_rng(234542)

    def test_small(self):

        for val in [0, 1]:
            self.assertTrue(np.allclose(intvert.invert_2D(signal=np.array([[val]])) - val, 0), "Trivial tests")

        for M, N in product(range(1, 10), range(1, 10)):

            with self.subTest(M=M, N=N):
                matrix = self.rand.integers(0, 2, (M, N))

                blurred = intvert.sample_2D(matrix)

                inverted = intvert.invert_2D(blurred)

                self.assertTrue(np.allclose(matrix - inverted, 0))

    def test_on_1D(self):

        def test_small(self):

            for N in range(1, 50):

                with self.subTest(N=N):
                    signal = self.rand.integers(0, 2, (N, 1))

                    blurred = intvert.sample_2D(signal)

                    inverted = intvert.invert_2D(blurred)

                    self.assertTrue(np.allclose(signal - inverted, 0), f"actual: {inverted}; expected: {signal}")

        def test_large_prime(self):

            for N in [53, 59]:

                with self.subTest(N=N):

                    signal = self.rand.integers(0, 2, (1, N))
                    blurred = intvert.sample_2D(signal)
                    inverted = intvert.invert_2D(blurred)
                    self.assertFalse(np.allclose(signal - inverted, 0), f"Too large for 1 coefficient, double precision")

                    with mp.get_context() as c:
                        c.precision = 200
                        blurred = intvert.sample_2D(signal)
                        inverted = intvert.invert_2D(blurred, beta2=1e20)
                        self.assertTrue(np.allclose(signal - inverted, 0), f"works with larger beta2; actual: {inverted}; expected: {signal}")

                    known_coeffs = intvert.select_coeffs_2D(1, N, [10])
                    blurred = intvert.sample_2D(signal, known_coeffs=known_coeffs)
                    inverted = intvert.invert_2D(blurred, known_coeffs=known_coeffs)
                    self.assertTrue(np.allclose(signal - inverted, 0), f"Works with several coefficients; actual: {inverted}; expected: {signal}")

        def test_coeffs(self):
            # runs all of the 1D tests through invert_2D

            known_coeffs = {
                (1, 6): {(1, 6): {frozenset({(0, 0)})}, (1, 1): {frozenset({(0, 5)})}, (1, 2): {frozenset({(0, 4)})}, (1, 3): {frozenset({(0, 3)})}},
                (12, 1): {
                    (12, 1): {frozenset({(0, 0)})}, 
                    (1, 1): {frozenset({(11, 0), (7, 0)})}, 
                    (2, 1): {frozenset({(10, 0)})}, 
                    (3, 1): {frozenset({(3, 0)})}, 
                    (4, 1): {frozenset({(8, 0)})}, 
                    (6, 1): {frozenset({(6, 0)})}
                },
                (1, 15): {
                    (1, 15): {frozenset({(0, 0)})}, 
                    (1, 1): {frozenset({(0, 14), (0, 13), (0, 11)})}, 
                    (1, 3): {frozenset({(0, 6)})}, 
                    (1, 5): {frozenset({(0, 5)})}
                },
                (30, 1): {
                    (30, 1): {frozenset({(0, 0)})}, 
                    (1, 1): {frozenset({(7, 0), (29, 0)})},
                    (2, 1): {frozenset({(8, 0), (14, 0)})}, 
                    (3, 1): {frozenset({(3, 0)})},
                    (5, 1): {frozenset({(25, 0)})}, 
                    (6, 1): {frozenset({(12, 0)})}, 
                    (10, 1): {frozenset({(10, 0)})}, 
                    (15, 1): {frozenset({(15, 0)})}
                },
            }
            for (M, N) in known_coeffs:
                with self.subTest(M=M, N=N):
                    for _ in range(10):
                        signal = self.rand.integers(0, 2, (M, N))
                        blurred = intvert.sample_2D(signal, known_coeffs=known_coeffs[M, N])
                        inverted = intvert.invert_2D(blurred, known_coeffs=known_coeffs[M, N])
                        self.assertTrue(np.allclose(signal - inverted, 0), f"works when given known coefficients; actual: {inverted}; expected: {signal}") 
                        inverted = intvert.invert_2D(blurred)
                        self.assertTrue(np.allclose(signal - inverted, 0), f"finds coefficients automatically; actual: {inverted}; expected: {signal}") 


        with self.subTest(test="test_small"):
            test_small(self)
        with self.subTest(test="large_prime"):
            test_large_prime(self)
        with self.subTest(test="test_coeffs"):
            test_coeffs(self)

    def test_large_prime(self):

        dimensions = [(7, 17), (3, 29)]

        for M, N in dimensions:
            with self.subTest(M=M, N=N):

                signal = self.rand.integers(0, 2, (M, N))
                blurred = intvert.sample_2D(signal)
                try:
                    inverted = intvert.invert_2D(blurred)
                    correct = np.allclose(signal - inverted, 0)
                except Exception:
                    correct = False
                finally:
                    self.assertFalse(correct, f"Too large for 1 coefficient, double precision")
                
                with mp.get_context() as c:
                    c.precision = 200
                    blurred = intvert.sample_2D(signal)
                    inverted = intvert.invert_2D(blurred, beta2=1e35, beta3=1e4)
                    self.assertTrue(np.allclose(signal - inverted, 0), f"works with larger beta2; actual: {inverted}; expected: {signal}")

                known_coeffs = intvert.select_coeffs_2D(M, N, [5])
                blurred = intvert.sample_2D(signal, known_coeffs=known_coeffs)
                inverted = intvert.invert_2D(blurred, known_coeffs=known_coeffs)
                self.assertTrue(np.allclose(signal - inverted, 0), f"Works with several coefficients; actual: {inverted}; expected: {signal}")


    def test_large_non_binary(self):

        for M, N in [(30, 30), (20, 30), (24, 24)]:

            with self.subTest(M=M, N=N):
                signal = self.rand.integers(0, 5, (M, N))
                blurred = intvert.sample_2D(signal)
                inverted = intvert.invert_2D(blurred)
                self.assertTrue(np.allclose(signal - inverted, 0), f"actual: {inverted}; expected: {signal}")


    def test_vectorize(self):

        M, N = 30, 15

        signals = self.rand.integers(0, 2, (10, M, N))
        blurred = intvert.sample_2D(signals)
        inverted = intvert.invert_2D(blurred)
        self.assertTrue(np.allclose(signals - inverted, 0))

        known_coeffs = intvert.select_coeffs_2D(M, N, [2])
        blurred = intvert.sample_2D(signals, known_coeffs)
        inverted = intvert.invert_2D(blurred, known_coeffs=known_coeffs, beta3=1e3)
        self.assertTrue(np.allclose(signals - inverted, 0))

