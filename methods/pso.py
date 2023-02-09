import matplotlib.axes
import numpy as np
import matplotlib.pyplot as plt

from cost_function import cost, pid_eval

# Process to be optimized
# Visualizing the loss landscape, 2D case
bounds = 5
x, y = np.array(np.meshgrid(np.linspace(-bounds, bounds, 100), np.linspace(-bounds, bounds, 100)))
X = np.array([x, y])
z = cost(X)
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

plt.figure(figsize=(8, 6))
plt.imshow(z, extent=[-bounds, bounds, -bounds, bounds])
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5)
contours = plt.contour(x, y, z, colors='black')

n_particles = 50
dimensions = 2

w = -0.8
w_damp = 0.8
w_min = -2
w_max = -0.1

iter_max = 5000

b_low = -bounds
b_upper = bounds

phi_p = 0.2
phi_g = 0.8

particle_position = -np.random.uniform(b_low, b_upper, [dimensions, n_particles])
temp_position = particle_position
best_position = np.array([np.inf, np.inf])


for k in range(particle_position.shape[1]):
    if cost(temp_position[:, k]) < cost(best_position):
        best_position = temp_position[:, k]

particle_velocity = np.random.uniform(-np.abs(b_upper - b_low), np.abs(b_upper - b_low), [dimensions, n_particles])

stable_thresh = 1000
k = 0
color = iter(plt.cm.plasma(np.linspace(0, 1, 200)))  # after 200 g_best points you lose.

for _ in range(iter_max):
    print(k)
    for i in range(n_particles):

        rho_p = np.random.random_sample(1)
        rho_g = np.random.random_sample(1)

        particle_velocity[:, i] = w * particle_velocity[:, i] + phi_p * rho_p * (temp_position[:, i] - particle_position[:, i]) + phi_g * rho_g * (best_position - particle_position[:, i])
        particle_position[:, i] = particle_position[:, i] + particle_velocity[:, i]

        if cost(particle_position[:, i]) < cost(temp_position[:, i]):
            temp_position[:, i] = particle_position[:, i]

        if cost(particle_position[:, i]) < cost(best_position):
            best_position = particle_position[:, i]

    w = w_min +(w_max - w_min) * (iter_max - k)/iter_max

    # print("iteration: ", k, "with a global best: ", cost(best_position))
    plt.plot(cost(best_position), 'x', c='red', markersize=1.05*k)
    plt.xlim([-bounds, bounds])
    plt.ylim([-bounds, bounds])
    plt.title([best_position, "with a minimum: ", cost(best_position)])
    plt.pause(0.005)
    print(cost(best_position))
    k += 1