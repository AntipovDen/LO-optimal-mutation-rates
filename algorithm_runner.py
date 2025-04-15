from one_plus_one_ea import OnePlusOneEA
from os.path import isfile
from multiprocessing import Pool

if __name__ == "__main__":
    runs = 10
    n = 128
    mut_rates_file = "data/lo/s1/rates_lo_opt_128"

    with open(mut_rates_file, 'r') as f:
        mut_rates = [[0] + [int(s) for s in reversed(line.split()) if s != '--'] for line in f.readlines() if '#' not in line]

    # for n in [2 ** i for i in range(3, 9)]:
    #     print("Running for n={}".format(n))
    #     mut_rates_file = "data/optimal_rates_{}".format(n)
    #
    #     if not isfile(mut_rates_file):
    #         remember_result(n)
    #
    # with open(mut_rates_file, 'r') as f:
    #     mut_rates = [[int(s) for s in line.split()] for line in f.readlines() if '#' not in line]

    # print(mut_rates)
    # exit(0)

    def run(seed):
        filename = "data/lo/s1/runs/lo_opt-n_{}-seed_{}".format(n, seed)
        OnePlusOneEA(seed, n, mut_rates).run(filename)

    with Pool(runs) as pool:
        pool.map(run, list(range(runs)))

