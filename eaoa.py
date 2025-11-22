import numpy as np
import math
import random
from tqdm import tqdm

class eaoa:
    def __init__(self, objective_func, dim, pop_size, max_iter, lb, ub, minimize=True):
        self.func = objective_func
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.minimize = minimize

        # scalar or array
        self.lb = np.array(lb) if isinstance(lb, (list, np.ndarray)) else np.full(dim, lb)
        self.ub = np.array(ub) if isinstance(ub, (list, np.ndarray)) else np.full(dim, ub)

        self.c1 = 2.1
        self.c2 = 5.6
        self.c3 = 1.95
        self.c4 = 0.65

        # eq 5
        self.x = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

        # initialize volume (vol), density (den), acceleration (acc)
        self.vol = np.random.rand(self.pop_size)
        self.den = np.random.rand(self.pop_size)
        self.acc = np.random.uniform(0, 1, (self.pop_size, self.dim))

        self.fitness = np.zeros(self.pop_size)

        # set initial best score based on optimizatio goal
        self.best_score = float('inf') if self.minimize else -float('inf')
        self.best_pos = np.zeros(self.dim)
        self.best_vol = 0
        self.best_den = 0
        self.best_acc = np.zeros(self.dim)

    # check if a new value is better than the current best
    def check_is_better(self, current_best, new_val):
        if self.minimize:
            return new_val < current_best
        else:
            return new_val > current_best

    def optimize(self):
        for i in range(self.pop_size):
            self.fitness[i] = self.func(self.x[i])
            if self.check_is_better(self.best_score, self.fitness[i]):
                self.best_score = self.fitness[i]
                self.best_pos = self.x[i].copy()
                self.best_vol = self.vol[i]
                self.best_den = self.den[i]
                self.best_acc = self.acc[i].copy()

        for t in tqdm(range(1, self.max_iter + 1), desc='EAOA Progress: '):
            # eq 8
            tf = math.exp((t - self.max_iter) /  self.max_iter)
            # density factor d (decrease overtime)
            d = math.exp((t - self.max_iter) / self.max_iter) - (t / self.max_iter)
            # eq 6
            r = np.random.rand()
            self.vol += r * (self.best_vol - self.vol)
            # eq 7
            self.den += r * (self.best_den - self.den)

            acc_temp = np.zeros_like(self.acc)
            for i in range(self.pop_size):
                if tf <= 0.5:
                    # eq 9 (exploration phase)
                    mr = random.randint(0, self.pop_size - 1)
                    acc_temp[i] = (self.den[mr] + self.vol[mr] * self.acc[mr]) / (self.den[i] * self.vol[i] + 1e-10)
                else:
                    # eq 9 (exploitation phase)
                    acc_temp[i] = (self.best_den + self.best_vol * self.best_acc) / (self.den[i] * self.vol[i] + 1e-10)

            # eq 10
            min_a, max_a = np.min(acc_temp), np.max(acc_temp)
            if max_a == min_a:
                acc_norm = np.zeros_like(acc_temp)
            else:
                acc_norm = 0.8 * ((acc_temp - min_a) / (max_a - min_a)) + 0.2
            self.acc = acc_norm

            # eq 17
            for i in range(self.pop_size):
                r1, r2, r3 = random.random(), random.random(), random.random()
                # eq 16
                beta = -0.5 + r1 * (0.5 - (-0.5))
                # eq 13
                p = 2 * r3 - self.c4
                # eq 14
                f_new = 1 * r2 if p <= 0.5 else -1 * r2
                t_var = self.c3 * tf

                if tf <= 0.5:
                    # eq 17 (exploration with opposite learning)
                    rand_idx = random.randint(0, self.pop_size - 1)
                    self.x[i] = self.x[i] + beta * self.c1 * r1 * self.acc[i] * d * (self.x[rand_idx] - self.x[i])
                else:
                    # eq 17 (exploitation with multiverse directing)
                    self.x[i] = self.best_pos + f_new * self.c2 * r1 * self.acc[i] * d * t_var * (self.best_pos - self.x[i])
                    # self.x[i] = self.best_pos + f_new * self.c2 * r1 * self.acc[i] * d * (t_var * self.best_pos - self.x[i])

                self.x[i] = np.clip(self.x[i], self.lb, self.ub)

            for i in range(self.pop_size):
                fit = self.func(self.x[i])
                if self.check_is_better(self.best_score, fit):
                    self.best_score = fit
                    self.best_pos = self.x[i].copy()
                    self.best_vol = self.vol[i]
                    self.best_den = self.den[i]
                    self.best_acc = self.acc[i].copy()

        return self.best_pos, self.best_score




