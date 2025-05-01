from math import log
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from probabilities import expected_total_runtime


def heatmap(filename, runs_prefix=None, with_numbers=False):
    font = {'size': 14}
    rc('font', **font)

    with open(filename, 'r') as f:
        matrix = np.array([np.array([float(s) if s != '--' else -1 for s in line.split()]) for line in f.readlines() if '#' not in line])

    mask = [[x < 0 for x in line] for line in matrix]
    heatmap_matrix = np.ma.array(matrix, mask=mask)

    plt.imshow(heatmap_matrix)
    plt.colorbar()
    plt.xlabel("OneMax")
    plt.ylabel("LeadingOnes")
    n = len(matrix)
    # numbers are only good for small heatmaps, up to n=16
    if with_numbers:
        matrix_max = max(max(matrix[lo][lo:n]) for lo in range(n))
        for lo in range(n):
            for om in range(lo, n):
                plt.text(om, lo, str(matrix[lo][om]), ha="center", va="center", color="w" if matrix[lo][om] < 0.75 * matrix_max else "k")

    # next line was used for highlighting the states with different optimal mutation rates for different states
    # data for that was taken from another project
    # plt.plot([7, 8, 5, 7], [0, 0, 3, 3], 'rX', markersize=16)
    if runs_prefix is not None:
        for seed in range(10):
            with open(runs_prefix.format(seed), 'r') as f:
                # remove [1:] from the next line, if the algorithms have not been re-run
                coordinates = [[int(s) for s in line.split()[1:]] for line in f.readlines()]
            y, x = zip(*coordinates)
            plt.plot(x, y, 'ro-')
    ax = plt.gca()
    ax.invert_yaxis()
    plt.savefig("figures/{}{}.pdf".format(filename.split('/', 1)[1], "-trajectories" if runs_prefix is not None else ""), bbox_inches='tight')
    plt.show()


def runtimes_plot():
    # names = ["s1/runtimes_all_ones_{}", "s1/runtimes_bin_approx_{}", "s1/runtimes_bruteforce_{}",
    #          "s1/runtimes_lo_opt_{}", "s2/runtimes_precise_{}"]
    # labels = ["All ones", "Bin approximation", "Brute force", "Optimal for \\LO", "$\\calStwo$ precise"]
    # names = ["s1/runtimes_lo_opt_{}", "s2/runtimes_opt_{}"]
    # labels = ["$\\calSone$-optimal", "$\\calStwo$-optimal"]
    # names = ["runtimes_precise_{}", "runtimes_portfolio_powers_{}", "runtimes_initial_interval_{}", "runtimes_evenly_spread_{}"]
    # labels = ["Full portfolio", "$\\{2^0, 2^1, 2^2, \\dots\\}$", "$\\{1, 2, 3\\}$", "$\\{1, \\frac{n}{3}, \\frac{2n}{3}\\}$"]
    names = ["runtimes_opt_{}", "runtimes_strict_{}"]
    labels = ["Standard selection", "Strict selection"]

    n_ranges = [[2 ** i for i in range(3, 9)] for _ in range(2)]
    colors = ["bo-", "ro-"]

    for name, label, ns , color in zip(names, labels, n_ranges, colors):
        values = []
        for n in ns:
            with open("data/lo/s2/" + name.format(n), 'r') as f:
                runtimes = np.array([np.array([float(s) if s != '--' else -1 for s in line.split()]) for line in f.readlines() if '#' not in line])
            t = expected_total_runtime(runtimes)
            values.append((n , t / n ** 2))
        print("\\addplot coordinates {{{}}};". format("".join("({},{})".format(x, y) for x, y in values)))
        print("\\addlegendentry{{{}}};".format(label))
        x, y = list(zip(*values))
        plt.plot(x, y, color, label=label)

    plt.legend(loc=4)
    plt.xlabel("Problem size $n$")
    plt.ylabel("T/n^2")
    plt.show()


if __name__ == "__main__":
    runtimes_plot()
    # heatmap("data/loom/s2/rates_portfolio_powers_128")
    # heatmap("data/loom/s2/runtimes_portfolio_powers_128")
    # heatmap("data/loom/s1/runtimes_all_ones_128", "data/loom/s1/runs/all_ones-n_128-seed_{}")

