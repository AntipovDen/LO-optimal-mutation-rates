import numpy as np
from scipy.stats import binom as bin
from probabilities import transition_probabilities, expected_total_runtime


# Function which computes the expected runtimes for each (lo, om) state when we optimize (lo, om) with a given
# policy of mutation rates based on the states. These rates are passed to this function as the second argument.
# Rates must be a list of size n and the i-th element is an array of size at least i+1 of integer numbers
# rate [i][j] is the rate for lo = n - i - 1 and om = n - j - 1
def calc_loom_s2_fixed_rates(n, rates):
    runtimes = [[0] * n for _ in range(n)]

    # To start with something, assuming that in this last state with lo=om=n-1 we always flip one bit.
    runtimes[0][0] = n

    for lo in reversed(range(n - 1)):
        for om in reversed(range(lo, n)):
            k = rates[n - lo - 1][n - om - 1]
            p = transition_probabilities(lo, om, n, k)
            p_leave = -p[n - lo - 1][n - om - 1] - sum(p[-1][n - om:])

            runtime_up_lo = sum(sum(p[n - new_lo - 1][n - new_om - 1] * runtimes[n - new_lo - 1][n - new_om - 1] for new_om in range(new_lo, n)) for new_lo in range(lo + 1, n))
            runtime_same_lo = sum(p[n - lo - 1][n - new_om - 1] * runtimes[n - lo - 1][n - new_om - 1] for new_om in range(om + 1, n))
            runtime = (1 + runtime_up_lo + runtime_same_lo) / p_leave
            runtimes[n - lo - 1][n - om - 1] = runtime

    # Adapting matrix for a heatmap so that indices corresponded to lo and om values
    # we want the lines to be LO values (in descending order), and each line to be OM values in ascending order,
    # that is, we need to reverse the elements in each lines, and also the lines order.
    runtimes = [line[::-1] for line in runtimes[::-1]]
    mask = [[1] * i + [0] * (n - i) for i in range(n)]
    return  np.ma.array(runtimes, mask=mask), np.ma.array([line[::-1] for line in rates[::-1]], mask=mask)


# Optimized function: (lo, om) aka "lexicographic selection" (we can only decrease OM value when we increase LO)
# State space: (lo, om), that is, full information we can get from fitness
# In this case we have a strict order on the states of the algorithm, and the algorithm never decreases its state,
# hence we can compute the runtimes and optimal k-s one-by-one iterating through the states in descending order.
# We can have a limited set of k-s that is given as the argument "portfolio", it must include one-bit flip
def calc_loom_s2(n, portfolio=None):
    if portfolio is None:
        portfolio = list(range(1, n + 1))

    # in the next two arrays an element [i][j] corresponds to lo = n - i - 1 and om = n - j - 1
    runtimes = [[0] * n for _ in range(n)]
    best_ks = [[0] * n for _ in range(n)]

    # to start with something
    runtimes[0][0] = n
    best_ks[0][0] = 1

    for lo in reversed(range(n - 1)):
        for om in reversed(range(lo, n)):
            # we are computing the expected runtime for the state (lo, om)
            best_runtime = float("inf")
            best_k = 0
            for k in portfolio:
                # computing the expected runtime for this state if we always flip k bits in it
                if k > n - lo:
                    break
                p = transition_probabilities(lo, om, n, k)
                p_leave = -p[n - lo - 1][n - om - 1] - sum(p[-1][n - om:])

                runtime_up_lo = sum(sum(p[n - new_lo - 1][n - new_om - 1] * runtimes[n - new_lo - 1][n - new_om - 1] for new_om in range(new_lo, n)) for new_lo in range(lo + 1, n))
                runtime_same_lo = sum(p[n - lo - 1][n - new_om - 1] * runtimes[n - lo - 1][n - new_om - 1] for new_om in range(om + 1, n))

                runtime = (1 + runtime_up_lo + runtime_same_lo) / p_leave
                if runtime < best_runtime:
                    best_runtime = runtime
                    best_k = k
            best_ks[n - lo - 1][n - om - 1] = best_k
            runtimes[n - lo - 1][n - om - 1] = best_runtime

    # adapting matrix for a heatmap so that indices corresponded to lo and om values:
    # we want the lines to be LO values (in descending order), and each line to be OM values in ascending order,
    # that is, we need to reverse the elements in each lines, and also the lines order.
    runtimes = [line[::-1] for line in runtimes[::-1]]
    best_ks = [line[::-1] for line in best_ks[::-1]]
    mask = [[int(k == 0) for k in line] for line in best_ks]
    return np.ma.array(runtimes, mask=mask), np.ma.array(best_ks, mask=mask)

# Compute optimal mutation policy and expected runtimes for each individual
# in the full bit-string state space S^(n) under lexicographic selection.
# Returns two dicts: T[x] is expected runtime from bit-tuple x,
# K[x] is the optimal k for x.
def calc_loom_sn(n, portfolio=None):
    if portfolio is None:
        portfolio = list(range(1, n+1))
    # generate all bit-strings of length n
    states = list(product([0,1], repeat=n))
    # precompute LO and OM for each state
    lo_vals = {x: sum(1 for i in range(n) if x[i]==1 and all(x[j]==1 for j in range(i)))
               for x in states}
    om_vals = {x: sum(x) for x in states}
    # lex order on (lo,om)
    sorted_states = sorted(states, key=lambda x: (lo_vals[x], om_vals[x]))
    # dynamic programming containers
    T = {}   # expected runtime
    K = {}   # best k
    # terminal: all-ones string
    ones = tuple([1]*n)
    T[ones] = 0
    K[ones] = 0
    # iterate in descending lex order (from hardest to easiest)
    for x in reversed(sorted_states[:-1]):
        lo, om = lo_vals[x], om_vals[x]
        best_t = np.inf
        best_k = None
        for k in portfolio:
            if k > n - lo:
                break
            num = 1.0
            leave_prob = 0.0
            # consider all possible offspring with better (lo,om)
            for y in states:
                if (lo_vals[y], om_vals[y]) <= (lo, om):
                    continue
                d = sum(a!=b for a,b in zip(x,y))
                if d != k:
                    continue
                pxy = 1.0/comb(n, k)
                leave_prob += pxy
                num += pxy * T[y]
            if leave_prob <= 0:
                continue
            t_val = num/leave_prob
            if t_val < best_t:
                best_t, best_k = t_val, k
        T[x] = best_t
        K[x] = best_k
    return T, K


# Optimized function: (lo, om) aka "lexicographic selection" (we can only decrease OM value when we increase LO)
# State space: lo, that is, we have a limited information about the state space
# In this case precise solution requires to go through all (n - 1)! possible combinations of k for each lo level,
# which is impossible. Hence, we approximate the best distribution by making an assumption on the probability
# of vising a state: the probability to enter level lo in state (lo, om) is Pr[Bin(n-lo-1, 1/2) = n - om - 1.
# This is a very rough approximation that is actually more suited to the case when we optimize LO (without OM),
# but anyways we are going to remove this part from the paper: it does not make sense to use only limited information.
# We can have a limited set of k-s that is given as the argument "portfolio", it must include one-bit flip
def calc_loom_s1_approx_binom(n, portfolio=None):
    if portfolio is None:
        portfolio = list(range(1, n + 1))

    # in the next two arrays an element [i][j] corresponds to lo = n - i - 1 and om = n - j - 1
    runtimes = [[0] * n for _ in range(n)]
    best_ks = [[1] * n for _ in range(n)]

    # to start with something
    runtimes[0][0] = n
    best_ks[0][0] = 1

    for lo in reversed(range(n - 1)):
        # approximate vector of distribution over the level states (it is not really true, but should be okay as
        # an approximation):
        distribution = bin.pmf(list(range(n - lo)), n - lo - 1, 0.5) # i-th element is Pr[Bin(n-lo-1, 0.5)=i]
        this_level_runtimes = [0.0] * (n - lo)
        this_level_k = 1
        best_expected_runtime = float("inf")

        # iterating through all possible k-s from the protfolio and finding the one which minimizes the expected runtime
        # conditional on our assumption on probabilities of which state we enter first
        for k in portfolio:
            if k > n - lo:
                break
            for om in reversed(range(lo, n)):
                p = transition_probabilities(lo, om, n, k)
                p_leave = -p[n - lo - 1][n - om - 1] - sum(p[-1][n - om:])
                runtime_higher_lo = sum(sum(
                    p[n - new_lo - 1][n - new_om - 1] * runtimes[n - new_lo - 1][n - new_om - 1] for new_om in
                    range(new_lo, n)) for new_lo in range(lo + 1, n))
                runtime_same_lo = sum(
                    p[n - lo - 1][n - new_om - 1] * this_level_runtimes[n - new_om - 1] for new_om in range(om + 1, n))
                this_level_runtimes[n - om - 1] = (1 + runtime_higher_lo + runtime_same_lo) / p_leave
            expected_runtime = sum(prob * t for prob, t in zip(distribution, this_level_runtimes))
            if expected_runtime < best_expected_runtime:
                best_expected_runtime = expected_runtime
                this_level_k = k
                runtimes[n - lo - 1][:n - lo] = this_level_runtimes[:]
        best_ks[n - lo - 1][:n - lo] = ([this_level_k] * (n - lo))[:]

    # adapting matrix for a heatmap so that indices corresponded to lo and om values:
    # we want the lines to be LO values (in descending order), and each line to be OM values in ascending order,
    # that is, we need to reverse the elements in each lines, and also the lines order.
    runtimes = [line[::-1] for line in runtimes[::-1]]
    best_ks = [line[::-1] for line in best_ks[::-1]]
    mask = [[int(t == 0) for t in line] for line in runtimes]
    return np.ma.array(runtimes, mask=mask), np.ma.array(best_ks, mask=mask)


# Optimized function: (lo, om) aka "lexicographic selection" (we can only decrease OM value when we increase LO)
# State space: lo, that is, we have a limited information about the state space
# This is the aforementioned precise computation of the best rates by iterating through all possible (n - 1)!
# combinations of k that we use for each lo level
# It requires a lot of computational resources, so we cannot do it for large n.
# Portfolio here is all possible k-s.
def calc_loom_s1_bruteforce(n):
    ks_vector = [1] * n  # i-th value is the value of k for lo = n - i - 1

    # The next function defines the order of the vectors of k in which we iterate through them
    def increase_ks_vector(ks_vector):
        for i in reversed(range(len(ks_vector))):
            if ks_vector[i] != i + 1:
                ks_vector[i] += 1
                for j in range(i + 1, len(ks_vector)):
                    ks_vector[j] = 1
                return

    best_k_vector = ks_vector.copy()
    rates = [[1] * (i + 1) + [0] * (n - i - 1) for i in range(n)]
    best_runtime = expected_total_runtime(calc_loom_s2_fixed_rates(n, rates)[0])

    while ks_vector < list(range(1, n + 1)):
        increase_ks_vector(ks_vector)
        rates = [[k] * (i + 1) + [0] * (n - i - 1) for i, k in enumerate(ks_vector)]
        t = expected_total_runtime(calc_loom_s2_fixed_rates(n, rates)[0])
        if t < best_runtime:
            best_k_vector = ks_vector.copy()
            best_runtime = t

    # adapting matrix for a heatmap so that indices corresponded to lo and om values:
    # we want the lines to be LO values (in descending order), and each line to be OM values in ascending order,
    rates = [[k] * (i + 1) + [0] * (n - i - 1) for i, k in enumerate(best_k_vector)]
    runtimes = calc_loom_s2_fixed_rates(n, rates)[0]
    rates = [line[::-1] for line in rates[::-1]]
    mask = [[1] * i + [0] * (n - i) for i in range(n)]
    return np.ma.array(runtimes, mask=mask), np.ma.array(rates, mask=mask)
