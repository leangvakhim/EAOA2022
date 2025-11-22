import numpy as np
import sys
import opfunu
from eaoa import eaoa

dim = 30
pop_size = 50
max_iter = 1000
lb = -100
ub = 100

# Function IDs 1 to 30
function_ids = range(1, 31)
results = {}

for f_id in function_ids:
    print(f"\n--- Running Function F{f_id} ---")

    try:
        func_name = f"F{f_id}2017"

        if not hasattr(opfunu.cec_based.cec2017, func_name):
                print(f"-> F{f_id}: Not found in library. Skipping.")
                continue

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

        results[f_id] = best_score
        print(f"-> F{f_id} Best Score: {best_score:.6e}")

    except Exception as e:
        print(f"-> F{f_id} Error: {e}")

# Final Summary
print(f"{'Func ID':<10} | {'Best Fitness':<20}")
print("-" * 35)
for f_id, score in results.items():
    print(f"F{f_id:<9} | {score:.6e}")