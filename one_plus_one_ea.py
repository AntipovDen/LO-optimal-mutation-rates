from random import Random
from enum import StrEnum


def leadingones(x):
    for i in range(len(x)):
        if x[i] == 0:
            return i
    return len(x)


def zeromax(x):
    return len(x) - sum(x)


class Selection(StrEnum):
    lo = "lo"
    loom = "loom"
    lo_strict = "lo-strict"
    # loom_strict = "loom-strict" # for future, possibly


# this algorithm is hardcoded to solve leadingones problem
class OnePlusOneEA:
    def __init__(self,
                 seed: int,
                 n: int,
                 mutation_rates_matrix: list[list[int]],
                 selection: Selection=Selection.lo,
                 init_x: list[int]=None):
        self.random = Random()
        self.random.seed(seed)
        self.n = n
        if init_x is not None:
            self.x = init_x
        else:
            self.x = [self.random.randint(0, 1) for _ in range(self.n)]
        self.mutation_rates = mutation_rates_matrix
        self.zm = zeromax(self.x)
        self.lo = leadingones(self.x)
        self.y = self.x.copy()
        if selection == Selection.lo:
            self.selection = self.selection_lo
        elif selection == Selection.loom:
            self.selection = self.selection_loom
        elif selection == Selection.lo_strict:
            self.selection = self.selection_lo_strict
        else:
            print("Wrong selection type, standard selection applied")
            self.selection = self.selection_lo


    # All selection methods compute the fitness of the bit string in self.y with the saved fitness values of self.x
    # and decide if they replace self.x with self.y.
    # These methods return True if there is a change in the state of the algorithm and False otherwise.
    def selection_lo(self):
        ylo = leadingones(self.y)
        if ylo >= self.lo:
            old_lo, old_zm = self.lo, self.zm
            self.x[:] = self.y[:]
            self.lo, self.zm = ylo, zeromax(self.y)
            return old_lo != self.lo or old_zm != self.zm
        return False

    def selection_loom(self):
        ylo = leadingones(self.y)
        yzm = zeromax(self.y)
        if ylo > self.lo or ylo == self.lo and yzm <= self.zm:
            old_lo, old_zm = self.lo, self.zm
            self.x[:] = self.y[:]
            self.lo, self.zm = ylo, yzm
            return old_lo != self.lo or old_zm != self.zm
        return False

    def selection_lo_strict(self):
        ylo = leadingones(self.y)
        if ylo > self.lo:
            self.x[:] = self.y[:]
            self.lo, self.zm = ylo, zeromax(self.y)
            return True # we definitely change at least the lo value
        return False

    # Returns True, if there was a change in the iteration (then we need to log the change)
    def iteration(self):
        # mutating self.x and saving it in self.y
        self.y[:] = self.x[:]
        k = self.mutation_rates[self.lo][self.zm]
        for i in self.random.sample(range(self.n), k):
            self.y[i] = 1 - self.y[i]

        return self.selection()

    def run(self, logfile):
        t = 1
        with open(logfile, 'w') as f:
            f.write("0 {} {}\n".format(self.lo, self.n - self.zm))
        while self.lo < self.n:
            if self.iteration():
                with open(logfile, 'a') as f:
                    f.write("{} {} {}\n".format(t, self.lo, self.n - self.zm))
            t += 1
