from eaoa import eaoa
# import cec2017.functions as cec2017
import numpy as np
from benchmark import (
    bent_cigar,
    sum_of_powers,
    zakharov,
    rosenbrock,
    f5_rastrigin,
    expanded_scaffer,
    lunacek_bi_rastrigin,
    non_continuous_rastrigin,
    levy,
    schwefel
)

function = bent_cigar
dim = 30
pop_size = 40
max_iter = 1000
lb = -100
ub = 100
time = 25
minimize = True
list_val = []

optimizer = eaoa(function, dim, pop_size, max_iter, lb, ub, minimize)
for _ in range(time):
    best_pos, best_score = optimizer.optimize()
    list_val.append(best_score)

mean_val = np.mean(list_val)
std_val = np.std(list_val)
min_val = np.min(list_val)
max_val = np.max(list_val)

print(f"Mean value: {mean_val:.4e}")
print(f"Standard deviation: {std_val:.4e}")
print(f"Minimum value: {min_val:.4e}")
print(f"Maximum value: {max_val:.4e}")
