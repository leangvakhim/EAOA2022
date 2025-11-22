import numpy as np
import sys
import opfunu
from eaoa import eaoa

dim = 30
pop_size = 50
max_iter = 1000
lb = -100
ub = 100
times = 25

# Function IDs 1 to 30
function_ids = range(1, 6)

final_stats = {}

for f_id in function_ids:
    func_name = f"F{f_id}2017"
    if not hasattr(opfunu.cec_based.cec2017, func_name):
        print(f"-> F{f_id}: Not found in library.")
        continue

    run_errors = []

    bias = f_id * 100

    for run in range(times):
        try:
            func_class = getattr(opfunu.cec_based.cec2017, func_name)
            benchmark_obj = func_class(ndim=dim)

            def objective_wrapper(x):
                return benchmark_obj.evaluate(x)

            optimizer = eaoa(
                objective_func=objective_wrapper,
                dim=dim,
                pop_size=pop_size,
                max_iter=max_iter,
                lb=lb,
                ub=ub,
                minimize=True
            )

            best_pos, best_score = optimizer.optimize()

            error = best_score - bias

            if abs(error) < 1e-8:
                error = 0.0

            run_errors.append(error)

            # print(f"  Run {run+1}/{times} - Error: {error:.6e}")

        except Exception as e:
            print(f"-> F{f_id} Run {run+1} Error: {e}")

    if run_errors:
        mean_val = np.mean(run_errors)
        std_val = np.std(run_errors)
        min_val = np.min(run_errors)
        max_val = np.max(run_errors)

        final_stats[f_id] = {
            'mean': mean_val,
            'std': std_val,
            'best': min_val,
            'worst': max_val
        }

        print(f"{'Func ID':<10} | {'Mean Error':<15} | {'Std Dev':<15} | {'Best Error':<15}")
        print("-" * 65)
        print(f"F{f_id:<9} | {mean_val:.4e}      | {std_val:.4e}      | {min_val:.4e}")

print("-" * 65)