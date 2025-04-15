from scipy.stats import hypergeom as hg, binom as bin


# returns the probability of having j leading ones in a random bit string of length m with exactly d zeros
# the formula for this probability is $\binom{m - j - 1, d - 1} / \binom{m, d}$, but here we optimize its
# computation, since computing binomial coefficients is expensive
def prob_of_freeriders(m, d, j):
    assert m >= d
    assert 0 <= j <= m - d  # actually, we should just return 0 if i > m - d
    if d == 0:
        return float(j == m)
    res = d / m
    for k in range(j):
        res *= (m - d - k)/(m - 1 - k)
    return res


# Returns an array of transition probabilities from state lo=i, om=j for problem size n and search radius k
# Returned p is such that element p[l][m] is the probability to go to state with lo = n - l - 1 and om = n - m - 1
# The indices ranges: l in [0..n - i - 1], m in [0..l]
# The returned array DOES NOT include the probability to go to the optimum in one step
# Note: p[n-i-1][n-j-1] is a negative of the probability to leave the state (or the probability to stay minus one),
# which takes into account the probability that we go to the optimum in one step
def transition_probabilities(i, j, n, k):
    assert i <= j <= n
    assert k + i <= n

    p = [[0] * (l + 1) for l in range(n - i)] # l is the inverse lo value, that is, l = n - lo - 1

    # moving in the same lo level: not flipping any bits in positions [0..i]
    if k + i < n:
        zeros_to_flip = list(range(max(0, k - (j - i)), min(k, n - j - 1) + 1))
        coeff1 = hg.pmf(0, n, i + 1, k)
        coeff2 = hg.pmf(zeros_to_flip, n - i - 1, n - j - 1, k)
        for z, c in zip(zeros_to_flip, coeff2): # z is the number of zeros in the tail that we flip
            # when we flip z zeros, we go to lo=i, om=j+2z-k, that is, l=n-i-1, m=n-(j+2z-k)-1
            new_om = j + 2 * z - k
            if new_om != j:
                p[n - i - 1][n - new_om - 1] = coeff1 * c
                p[n - i - 1][n - j - 1] -= p[n - i - 1][n - new_om - 1]

    # moving to a better lo level: not flipping bits in positions [0..i-1], but flipping the bit in position i
    zeros_to_flip = list(range(max(0, k - 1 - (j - i)), min(k - 1, n - j - 1) + 1))
    coeff1 = hg.pmf(1, n, i + 1, k) / (i + 1)
    coeff2 = hg.pmf(zeros_to_flip, n - i - 1, n - j - 1, k - 1)
    for z, c in zip(zeros_to_flip, coeff2): # z is again the number of bits flipped
        # here we go to some state that has a particular om value, namely om=j+2z-k
        # its lo value might differ: it is i+1+(# of free-riders)
        # so we iterate through all possible number of those free-riders
        new_om = j + 2 * z - (k - 1) + 1
        coeff3 = coeff1 * c
        if new_om < n: # otherwise we do not need to remember this probability
            for fr in range(new_om - (i + 1) + 1):
                new_lo = i + 1 + fr
                p[n - new_lo - 1][n - new_om - 1] = coeff3 * prob_of_freeriders(n - i - 1, n - new_om, fr)
        # subtracting the probability to leave the current state
        p[n - i - 1][n - j - 1] -= coeff3

    return p


# Given an array of runtimes for different states, this function returns a single number, which is the total
# expected runtime over all possible initial states (computing it by the law of total expectation)
def expected_total_runtime(runtimes):
    n = len(runtimes)
    runtime = 0.0
    for om, c in zip(range(n), bin.pmf(list(range(n)), n, 0.5)):
        for free_riders in range(om):
            runtime += c * prob_of_freeriders(n, n - om, free_riders) * runtimes[free_riders][om]
    return runtime