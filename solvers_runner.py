from multiprocessing import Pool

from matplotlib import pyplot as plt
from math import floor
from loom_solver import *
from lo_solver import *


def run_and_show_heatmaps(function, n, rates=None):
    if rates is None:
        T, K = function(n)
    else:
        T, K = function(n, rates)

    plt.imshow(K)
    for lo in range(n):
        for om in range(lo, n):
            plt.text(om, lo, K[lo][om], ha="center", va="center", color="w" if K[lo][om] < 0.75 * n else "k")
    plt.colorbar()
    plt.xlabel("OneMax")
    plt.ylabel("LeadingOnes")
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()

    plt.imshow(T)
    plt.colorbar()
    plt.xlabel("OneMax")
    plt.ylabel("LeadingOnes")
    ax = plt.gca()
    ax.invert_yaxis()
    plt.show()


# running and writing specific settings

# (LO, OM) function
def loom_s2_precise(n):
    T, K = calc_loom_s2(n)
    with open("data/loom/s2/rates_precise_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/loom/s2/runtimes_precise_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


def loom_s1_bruteforce(n):
    T, K = calc_loom_s1_bruteforce(n)
    with open("data/loom/s1/rates_bruteforce_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/loom/s1/runtimes_bruteforce_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


def loom_s1_bin_approx(n):
    T, K = calc_loom_s1_approx_binom(n)
    with open("data/loom/s1/rates_bin_approx_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/loom/s1/runtimes_bin_approx_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


def loom_s1_all_ones(n):
    rates = [[1] * n for _ in range(n)]
    T, K = calc_loom_s2_fixed_rates(n, rates)
    with open("data/loom/s1/rates_all_ones_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/loom/s1/runtimes_all_ones_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


def loom_s1_lo_opt(n):
    rates = [[floor(n / (n - i))] * n for i in range(n)]
    T, K = calc_loom_s2_fixed_rates(n, rates)
    with open("data/loom/s1/rates_lo_opt_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/loom/s1/runtimes_lo_opt_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")
            
            
# LO function
def lo_s1_lo_opt(n):
    rates = [[floor(n / (n - i))] * n for i in range(n)]
    T, K = calc_lo_s2_fixed_rates(n, rates)
    with open("data/lo/s1/rates_lo_opt_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/lo/s1/runtimes_lo_opt_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")
            

def lo_s1_all_ones(n):
    rates = [[1] * n for _ in range(n)]
    T, K = calc_lo_s2_fixed_rates(n, rates)
    with open("data/lo/s1/rates_all_ones_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/lo/s1/runtimes_all_ones_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


def lo_s2_relaxations(n):
    T, K = calc_lo_s2_relaxations(n)
    with open("data/lo/s2/rates_precise_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/lo/s2/runtimes_precise_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


def lo_s2_strict(n):
    T, K = calc_lo_s2_strict(n)
    with open("data/lo/s2/rates_strict_{}".format(n), 'w') as f:
        for line in K:
            f.write(" ".join(str(k) for k in line))
            f.write("\n")
    with open("data/lo/s2/runtimes_strict_{}".format(n), 'w') as f:
        for line in T:
            f.write(" ".join(str(t) for t in line))
            f.write("\n")


# Function for multiprocessing run
def run(args):
    args[0](args[1])


if __name__ == "__main__":
    # we want to run:
    # loom_s2_precise(n) for n in 4, 8, 16, ... 512
    # lo_s2_relaxations(n) for n in 4, 8, 16, ... 512
    # lo_s1_all_ones(n) for n in 4, 8, 16, ... 512
    # lo_s1_lo_opt(n) for n in 4, 8, 16, ... 512
    # lo_s2_strict for n in 4, 8, 16, ... 512

    # we do not want anymore to run:
    # loom_s1_brute_force(n) for n in 4,6,8 (?)
    # loom_s1_bin_approx(n) for n in 4, 8, 16, ... 512
    # loom_s1_all_ones(n) for n in 4, 8, 16, ... 512
    # loom_s1_lo_opt(n) for n in 4, 8, 16, ... 512


    loom_s2_precise_ns = [2 ** i for i in range(2, 10)]
    loom_s2_precise_jobs = [(loom_s2_precise, n) for n in loom_s2_precise_ns]
    # loom_s1_bruteforce_ns = [4, 6, 8]
    # loom_s1_bruteforce_jobs = [(loom_s1_bruteforce, n) for n in loom_s1_bruteforce_ns]
    # loom_s1_bin_approx_ns = [2 ** i for i in range(8, 10)]
    # loom_s1_bin_approx_jobs = [(loom_s1_bin_approx, n) for n in loom_s1_bin_approx_ns]
    # loom_s1_all_ones_ns = [2 ** i for i in range(10, 11)]
    # loom_s1_all_ones_jobs = [(loom_s1_all_ones, n) for n in loom_s1_all_ones_ns]
    # loom_s1_lo_opt_ns = [2 ** i for i in range(9, 11)]
    # loom_s1_lo_opt_jobs = [(loom_s1_lo_opt, n) for n in loom_s1_lo_opt_ns]
    lo_s2_relaxations_ns = [2 ** i for i in range(2, 10)]
    lo_s2_relaxations_jobs = [(lo_s2_relaxations, n) for n in lo_s2_relaxations_ns]
    lo_s2_strict_ns = [2 ** i for i in range(8, 10)]
    lo_s2_strict_jobs = [(lo_s2_strict, n) for n in lo_s2_strict_ns]
    lo_s1_all_ones_ns = [2 ** i for i in range(2, 10)]
    lo_s1_all_ones_jobs = [(lo_s1_all_ones, n) for n in lo_s1_all_ones_ns]
    lo_s1_lo_opt_ns = [2 ** i for i in range(2, 10)]
    lo_s1_lo_opt_jobs = [(lo_s1_lo_opt, n) for n in lo_s1_lo_opt_ns]



    # total_len = len(loom_s2_precise_ns) + len(lo_s2_relaxations_ns) + len(lo_s1_all_ones_ns) + len(lo_s1_lo_opt_ns)
    # + len(loom_s1_bruteforce_ns) + len(loom_s1_bin_approx_ns) + len(loom_s1_all_ones_ns) + len(loom_s1_lo_opt_ns)
    total_len = len(lo_s2_strict_ns)

    # all_arguments = loom_s2_precise_jobs + lo_s2_relaxations_jobs + lo_s1_all_ones_jobs + lo_s1_lo_opt_jobs
    # + loom_s1_bruteforce_jobs + loom_s1_bin_approx_jobs + loom_s1_all_ones_jobs + loom_s1_lo_opt_jobs
    all_arguments = lo_s2_strict_jobs
    with Pool(total_len) as pool:
        pool.map(run, all_arguments)

