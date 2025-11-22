import numpy as np
import sys
from eaoa import eaoa
import cec2017.functions as cec

dim = 30
pop_size = 50
max_iter = 1000
lb = -100
ub = 100

function_ids = range(1, 31)

results = {}

for f_id in function_ids:
    print(f"\n--- Running Function F{f_id} ---")

    try:
        def objective_wrapper(x):
            return cec.all_functions[f_id](np.array([x]))[0]

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

        # 4. Store and Print
        results[f_id] = best_score
        print(f"-> F{f_id} Best Score: {best_score:.6e}")

    except Exception as e:
        print(f"-> F{f_id} Skipped or Error: {e}")

# Final Summary
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"{'Func ID':<10} | {'Best Fitness':<20}")
print("-" * 35)
for f_id, score in results.items():
    print(f"F{f_id:<9} | {score:.6e}")
