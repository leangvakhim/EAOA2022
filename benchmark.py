import numpy as np

def sphere_func(x):
    return np.sum(x**2)

# 1. Unimodal Functions (F1 - F3)

# F1: Bent Cigar Function
def bent_cigar(x):
    return x[0]**2 + 10**6 * np.sum(np.square(x[1:]))

# F2: Sum of Different Powers
def sum_of_powers(x):
    d = len(x)
    idx = np.arange(1, d + 1)
    return np.sum(np.abs(x) ** (idx + 1))

# F3: Zakharov Function
def zakharov(x):
    d = len(x)
    sum1 = np.sum(np.square(x))
    sum2 = np.sum(0.5 * np.arange(1, d + 1) * x)
    return sum1 + sum2**2 + sum2**4

# F4: Rosenbrock's Function
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# F5: Rastrigin's Function
def f5_rastrigin(x):
    return np.sum(np.square(x) - 10 * np.cos(2 * np.pi * x) + 10)

# F6: Expanded Scaffer's
def expanded_scaffer(x):
    def scaffer_term(x_i, x_next):
        term1 = np.sin(np.sqrt(x_i**2 + x_next**2))**2 - 0.5
        term2 = (1 + 0.001 * (x_i**2 + x_next**2))**2
        return 0.5 + term1 / term2
    d = len(x)
    val = 0
    for i in range(d - 1):
        val += scaffer_term(x[i], x[i+1])
    val += scaffer_term(x[d-1], x[0]) # Wrap around
    return val

# F7: Lunacek Bi-Rastrigin
def lunacek_bi_rastrigin(x):
    d = len(x)
    mu0 = 2.5
    s = 1 - 1 / (2 * np.sqrt(d + 20) - 8.2)
    mu1 = -np.sqrt((mu0**2 - 1) / s)

    term1 = np.sum((x - mu0)**2)
    term2 = np.sum((x - mu1)**2)
    term3 = d * 10 + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)) # Rastrigin part

    return min(term1, d * 1.0 + s * term2) + term3

# F8: Non-Continuous Rastrigin
def non_continuous_rastrigin(x):
    y = np.where(np.abs(x) >= 0.5, x, np.round(2 * x) / 2)
    return np.sum(np.square(y) - 10 * np.cos(2 * np.pi * y) + 10)

# F9: Levy Function
def levy(x):
    d = len(x)
    w = 1 + (x - 1) / 4

    term1 = np.sin(np.pi * w[0])**2
    term3 = (w[d-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[d-1])**2)

    sum_mid = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))

    return term1 + sum_mid + term3

# F10: Schwefel's Function
def schwefel(x):
    d = len(x)
    return 418.9829 * d - np.sum(x * np.sin(np.sqrt(np.abs(x))))



