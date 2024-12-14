import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import hypergeom as hg


def prob_of_freeriders(m, d, j):
    # returns the probability of having j leading ones in a random bit string of length m with exactly d zeros
    assert m >= d
    assert 0 <= j <= m - d  # actually, we should just return 0 if i > m - d
    if d == 0:
        return float(j == m)
    res = d / m
    for k in range(j):
        res *= (m - d - k)/(m - 1 - k)
    return res


def solve_for_one_bit_flips(n):
    # todo: make it consistent with the new function below
    expected_runtimes = [[0] * n for i in range(n)]  # the states are defined by lo, (zm - 1)

    expected_runtimes[-1][0] = n

    for m in range(2, n + 1):  # m is n-lo(x)
        # print("m={}".format(m))
        aa = [[0] * m for _ in range(m)]
        aa[0][0], aa[0][1] = m/n, -(m - 1)/n
        for d in range(2, m):  # d is zm(x), it is from [1..m]
            aa[d - 1][d - 2], aa[d - 1][d - 1], aa[d - 1][d] = -(d - 1)/n, m/n, -(m - d)/n
        aa[m - 1][m - 2], aa[m - 1][m - 1] = -(m - 1)/n, m/n
        aa = np.array(aa)

        bb = [1] * m
        for d in range(2, m + 1):
            bb[d - 1] += sum([prob_of_freeriders(m - 1, d - 1, i) * expected_runtimes[-(m - 1 - i)][d - 2] for i in range(m - d + 1)]) / n
        bb = np.array(bb)

        x = np.linalg.solve(aa, bb)
        expected_runtimes[-m][:m] = x[:]
    return expected_runtimes


# portfolio is a set of numbers of bits we can flip
def solve_iteratively(n, portfolio=None):
    if portfolio is None:
        portfolio = list(range(1, n + 1))
    expected_runtimes = [[0] * (n + 1) for _ in range(n + 1)]  # the states are defined by m = n - lo and d = n - om
    optimal_bits_to_flip = [[1] * (n + 1) for _ in range(n + 1)]

    expected_runtimes[1][1] = n

    for m in range(2, n + 1):  # m is n-lo(x)
        print("m={}".format(m))

        # pre-computing some parts which we will use a lot
        b = [[1] * (m + 1) for _ in range(m)] # vector i corresponds to state d = i + 1 in [1..m]
        for k in portfolio: # compute b[*][k]
            if k <= m:
                for d in range(1, m + 1):
                    zeros_to_flip = list(range(max(0, d + k - 1 - m), min(d - 1, k - 1) + 1))
                    hg_coeffs_1 = hg.pmf(zeros_to_flip, m - 1, d - 1, k - 1)
                    hg_coeff_2 = hg.pmf(k - 1, n, m - 1, k) / (n - m + 1)
                    for i, coeff in zip(zeros_to_flip, hg_coeffs_1):
                        new_d = d + k - 2 * (i + 1)
                        b[d - 1][k] += coeff * hg_coeff_2 * sum(
                            prob_of_freeriders(m - 1, new_d, j) * expected_runtimes[m - 1 - j][new_d] for j in
                            range(m - new_d))

        # transition probabilities between states, which are used then in a system of linear equations
        # we want a[i][j][k] be the transition probability from state d = i + 1 to state d = j + 1 when we flip k bits
        a = [[[0] * (m + 1) for j in range(m)] for i in range(m)]
        for k in portfolio:
            if k <= m:
                for d in range(1, m + 1): # d is the number of zero-bits
                    # setting coefficients for the system of linear equations when in state (m, d) we flip k bits
                    # their absolute values are also transition probabilities between these states
                    zeros_to_flip = list(range(max(0, d + k - m), min(d - 1, k) + 1))
                    hg_coeffs_1 = hg.pmf(zeros_to_flip, m - 1, d - 1, k)
                    hg_coeff_2 = hg.pmf(k, n, m - 1, k)
                    for i, coeff in zip(zeros_to_flip, hg_coeffs_1): # i is the number of zero-bits we flip (hence, getting to the state d - i + (k - i))
                        a[d - 1][d + k - 2 * i - 1][k] = -coeff * hg_coeff_2
                    a[d - 1][d - 1][k] += hg.pmf(k, n, m, k)

        # first solve the system assuming we always flip one bit in all states
        a1 = np.array([[column[1] for column in row] for row in a])
        b1 = np.array([x[1] for x in b])

        x = np.linalg.solve(a1, b1)
        expected_runtimes[m][1:m + 1] = x[:]

        # now we are relaxing it by choosing different mutation rates from portfolio until something changes
        new_opt_rate_flag = True
        while new_opt_rate_flag:
            print("relaxation")
            new_opt_rate_flag = False

            # find the new optimal rates assuming that other states have the same
            for d in range(1, m + 1):
                optimal_rate, optimal_runtime = optimal_bits_to_flip[m][d], expected_runtimes[m][d]
                for k in portfolio:
                    if k > m:  # such a rate cannot put us to another state, hence cannot be optimal
                        continue
                    new_runtime = b[d - 1][k]

                    for i in range(max(0, d + k - m), min(d - 1, k) + 1):
                        new_state = d + k - 2 * i
                        if new_state != d:
                            new_runtime += -a[d - 1][new_state - 1][k] * expected_runtimes[m][new_state]

                    new_runtime /= a[d - 1][d - 1][k]
                    if new_runtime < optimal_runtime:
                        optimal_rate, optimal_runtime = k, new_runtime
                if optimal_rate != optimal_bits_to_flip[m][d]:
                    new_opt_rate_flag = True
                    optimal_bits_to_flip[m][d] = optimal_rate

            # re-solve the system with the new rates to update the true runtimes
            # (the ones we computed are just approximations!)
            ak = np.array([[a[i][j][optimal_bits_to_flip[m][i + 1]] for j in range(m)] for i in range(m)])
            bk = np.array([b[i][optimal_bits_to_flip[m][i + 1]] for i in range(m)])

            x = np.linalg.solve(ak, bk)
            for new_t, old_t in zip(x, expected_runtimes[m][1:m + 1]):
                if new_t > old_t:
                    print("Relaxation worsen the runtime :(")
            expected_runtimes[m][1:m + 1] = x[:]

    return expected_runtimes, optimal_bits_to_flip


# T = solve_for_one_bit_flips(32)
# for line in T:
#     print(len(line), line)
# plt.imshow(T)
# plt.colorbar()
# plt.xlabel("n - OneMax")
# plt.ylabel("n - LeadingOnes")
# plt.show()

T, K = solve_iteratively(64)
for line in T:
    print(len(line), line)
plt.imshow(T)
plt.colorbar()
plt.xlabel("n - OneMax")
plt.ylabel("n - LeadingOnes")
plt.show()
for line in K:
    print(len(line), line)
plt.imshow(K)
plt.colorbar()
plt.xlabel("n - OneMax")
plt.ylabel("n - LeadingOnes")
plt.show()