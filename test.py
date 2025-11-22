import numpy as np
from eaoa import eaoa
import matplotlib.pyplot as plt

w = 100
l = 100
num_sensors = 50
rs = 10
pop_size = 40
max_iter = 100
minimize = False


grid_res = 1.0
grid_x = np.arange(0, w, grid_res)
grid_y = np.arange(0, l, grid_res)
gx, gy = np.meshgrid(grid_x, grid_y)
grid_points = np.column_stack((gx.ravel(), gy.ravel()))
total_points = len(grid_points)

def wsn_objective(position_vector):
    nodes = position_vector.reshape((num_sensors, 2))
    dists = np.linalg.norm(grid_points[:, np.newaxis, :] - nodes[np.newaxis, :, :], axis=2)
    detection_probs = (dists < rs).astype(float)
    is_covered = np.any(detection_probs, axis=1)
    return np.sum(is_covered) / total_points

dim = num_sensors * 2
lb = np.zeros(dim)
ub = np.zeros(dim)
ub[0::2] = w
ub[1::2] = l

optimizer = eaoa(wsn_objective, dim, pop_size, max_iter, lb, ub, minimize)

best_pos, best_score = optimizer.optimize()

# print(f"Best position: {best_pos}")
print(f"Coverage: {best_score * 100:.2f}")

best_coords = best_pos.reshape((num_sensors, 2))

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, w)
ax.set_ylim(0, l)
ax.set_aspect('equal')

for i in best_coords:
    circle = plt.Circle((i[0], i[1]), rs, color='r', alpha=0.2)
    ax.add_artist(circle)
    ax.plot(i[0], i[1], 'k.', markersize=4)

plt.title(f"Deployment (coverage: {best_score*100 :.2f})")
plt.xlabel('Width (m)')
plt.ylabel('Height (m)')
plt.grid(True, linestyle='--')
plt.show()

