import numpy as np
from probabilities import transition_probabilities


# Helper function which computes the expected runtimes for each (lo, om) state when we optimize LO with a given
# policy of mutation rates based on the states. These rates are passed to this function as the second argument.
# Rates must be a list of size n and the i-th element is an array of size at least i+1 of integer numbers
# rate [i][j] is the rate for lo = n - i - 1 and om = n - j - 1
def calc_lo_s2_fixed_rates(n, rates):
    # Rates must be a list of size n and the i-th element is an array of size at least i+1 of integer numbers
    # Rate [i][j] is the rate for lo = n - i - 1 and om = n - j - 1
    runtimes = [[0] * n for _ in range(n)]  # indexed with [n-lo-1][n-om-1]
    runtimes[0][0] = n

    for lo in reversed(range(n - 1)):  # m is n-lo(x)
        # print("m={}".format(m))

        # getting transition_probabilities for this value:
        trans_probs = [transition_probabilities(lo, om, n, rates[n - lo - 1][n - om - 1]) for om in range(n - 1, lo - 1, -1)]

        # components for the system of equations
        a = np.array([[-trans_probs[n - om - 1][n - lo - 1][n - new_om - 1] for new_om in range(n - 1, lo - 1, -1)] for om in reversed(range(lo, n))])
        b = [1 + sum(sum(trans_probs[n - om - 1][n - new_lo - 1][n - new_om - 1] * runtimes[n - new_lo - 1][n - new_om - 1] for new_om in range(new_lo, n)) for new_lo in range(lo + 1, n)) for om in reversed(range(lo, n))]
        x = np.linalg.solve(a, b)
        runtimes[n - lo - 1][:n - lo] = x[:]

    runtimes = [line[::-1] for line in runtimes[::-1]]
    mask = [[1] * i + [0] * (n - i) for i in range(n)]
    return  np.ma.array(runtimes, mask=mask), np.ma.array([line[::-1] for line in rates[::-1]], mask=mask)


# Optimized function: LO aka "standard selection" (we can decrease OM, but cannot decrease LO)
# State space: (lo, om), that is, we have more information about the state than just fitness
# Precise solution requires checking all possible combinations of rates for the states space, which is impossible:
# there are like Prod_{i = 1}^n i^i combinations (that is, more than n^n).
# For this reason we are computing the "optimal" rates level-by-level, starting with the top LO level.
# For each level we take a vector of k-s (initially, all-ones vector) and estimate the expected runtimes for each state
# with those k-s. Then with these estimates for each state we are finding k that minimizes the runtime from this state
# assuming our estimates of runtimes for other states. Then we get a new vector of k-s and compute new estimates.
# We repeat this relaxation while at least one value of k is changed.
# We can have a limited set of k-s that is given as the argument "portfolio", it must include one-bit flip,
# which is the first value in this portfolio. The rest of the rates must be in ascending order
def calc_lo_s2_relaxations(n, portfolio=None):
    if portfolio is None:
        portfolio = list(range(1, n + 1))

    runtimes = [[0] * n for _ in range(n)]  # indexed with [n-lo-1][n-om-1]
    best_k_indices = [[0] * n for _ in range(n)] # we need to keep indices but not values for the easier access to
                                                 # transition probabilities
    runtimes[0][0] = n

    for lo in reversed(range(n - 1)):
        # Getting transition_probabilities for this value
        # trans_probs[i][n - om - 1][n - new_lo - 1][n - new_om - 1] contains the probability to go from state (lo, om)
        # to state (new_lo, new_om) when we flip exactly k bits, with only exception for (new_lo, new_om) = (lo, om):
        # in this case it contains a negative of the probability to leave the state.
        trans_probs = [[transition_probabilities(lo, om, n, k) for om in reversed(range(lo, n))] for k in portfolio if k <= n - lo]

        # Pre-computing vector b which we use to solve Ax = b for all possible k-s
        # bk[i][n - om - 1] is the element of vector b which corresponds to state (lo, om) when we use mutation rate
        # portfolio[i]
        bk = [[1 + sum(sum(trans_probs[i][n - om - 1][n - new_lo - 1][n - new_om - 1] * runtimes[n - new_lo - 1][n - new_om - 1] for new_om in range(new_lo, n)) for new_lo in range(lo + 1, n)) for om in reversed(range(lo, n))] for i, k in enumerate(portfolio) if k <= n - lo]

        # First we solve the system assuming we always flip one bit in all states
        a = np.array([[-trans_probs[0][n - om - 1][n - lo - 1][n - new_om - 1] for new_om in reversed(range(lo, n))] for om in reversed(range(lo, n))])
        b = np.array(bk[0])
        x = np.linalg.solve(a, b)
        runtimes[n - lo - 1][:n - lo] = x[:]

        # Now we are relaxing it by choosing different mutation rates from portfolio until something changes
        new_opt_rate_flag = True
        while new_opt_rate_flag:
            new_opt_rate_flag = False # if it is flipped to True, it means that in this relaxation we have changed
                                      # a rate of at least one state

            # Find the new optimal rates assuming that other states do not change their runtimes because of it.
            for om in range(lo, n):
                optimal_rate_index = best_k_indices[n - lo - 1][n - om - 1]
                optimal_runtime = runtimes[n - lo - 1][n - om - 1]
                for ki, k in enumerate(portfolio):
                    if k > n - lo:  # such a rate cannot put us to another state, hence cannot be optimal
                        break
                    new_runtime = bk[ki][n - om - 1] # this term includes the contribution of transitions to all the
                                                     # states with strictly larger LO to the expected runtime

                    # computing the contribution of the transitions to the same LO level
                    for zeros_flipped in range(max(0, k - (om - lo)), min(n - om - 1, k) + 1):
                        new_om = om + 2 * zeros_flipped - k
                        if new_om != om:
                            new_runtime += trans_probs[ki][n - om - 1][n - lo - 1][n - new_om - 1] * runtimes[n - lo - 1][n - new_om - 1]

                    new_runtime /= -trans_probs[ki][n - om - 1][n - lo - 1][n - om - 1] # dividing by the probability
                                                                                        # to leave the current state
                    if new_runtime < optimal_runtime:
                        optimal_rate_index, optimal_runtime = ki, new_runtime
                if optimal_rate_index != best_k_indices[n - lo - 1][n - om - 1]:
                    new_opt_rate_flag = True # marking that we have changed something, hence we need one more relaxation
                    best_k_indices[n - lo - 1][n - om - 1] = optimal_rate_index

            # Re-solve the system with the new rates to update the true runtimes
            # (the ones we computed are just approximations!)
            a = np.array([[-trans_probs[best_k_indices[n - lo - 1][n - om - 1]][n - om - 1][n - lo - 1][n - new_om - 1] for new_om in range(n - 1, lo - 1, -1)] for om in reversed(range(lo, n))])
            b = np.array([bk[best_k_indices[n - lo - 1][n - om - 1]][n - om - 1] for om in reversed(range(lo, n))])
            x = np.linalg.solve(a, b)
            runtimes[n - lo - 1][:n - lo] = x[:]

    runtimes = [line[::-1] for line in runtimes[::-1]]
    best_ks = [[portfolio[i] for i in line[::-1]] for line in best_k_indices[::-1]]
    mask = [[int(k == 0) for k in line] for line in runtimes]
    return  np.ma.array(runtimes, mask=mask),  np.ma.array(best_ks, mask=mask)


# Optimized function: LO with strict selection (we can decrease OM, but only when we increase LO)
# State space: (lo, om), that is, we have more information about the state than just fitness
# We have an order of the states, so we can compute the exact expected runtimes for any give rate state-by-state.
# We can have a limited set of k-s that is given as the argument "portfolio", it must include one-bit flip,
# which is the first value in this portfolio. The rest of the rates must be in ascending order
def calc_lo_s2_strict(n, portfolio=None):
    if portfolio is None:
        portfolio = list(range(1, n + 1))

    runtimes = [[float("inf")] * n for _ in range(n)]  # indexed with [n-lo-1][n-om-1]
    best_ks = [[0] * n for _ in range(n)] # in this case we can just keep values
    runtimes[0][0] = n
    best_ks[0][0] = 1

    for lo in reversed(range(n - 1)):
        print(n, lo)
        for om in range(lo, n):
            for k in portfolio:
                if k > n - lo: # larger k are guaranteed not to increase lo value
                    break

                # getting the transition probabilities for the current state
                trans_probs = transition_probabilities(lo, om, n, k)
                p_leave = -sum(trans_probs[n - lo - 1])

                runtime_up_lo = sum(sum(
                    trans_probs[n - new_lo - 1][n - new_om - 1] * runtimes[n - new_lo - 1][n - new_om - 1] for new_om in
                    range(new_lo, n)) for new_lo in range(lo + 1, n))
                runtime = (1 + runtime_up_lo) / p_leave
                if runtime < runtimes[n - lo - 1][n - om - 1]:
                    runtimes[n - lo - 1][n - om - 1] = runtime
                    best_ks[n - lo - 1][n - om - 1] = k

    # adapting matrix for a heatmap so that indices corresponded to lo and om values:
    # we want the lines to be LO values (in descending order), and each line to be OM values in ascending order,
    # that is, we need to reverse the elements in each lines, and also the lines order.
    runtimes = [line[::-1] for line in runtimes[::-1]]
    best_ks = [line[::-1] for line in best_ks[::-1]]
    mask = [[int(k == 0) for k in line] for line in best_ks]
    return np.ma.array(runtimes, mask=mask), np.ma.array(best_ks, mask=mask)