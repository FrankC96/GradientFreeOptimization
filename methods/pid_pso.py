import numpy as np
import matplotlib.pyplot as plt

from cost_function import pid_eval as cost

# Visualizing the loss landscape, 2D case
Cost = 'pid'

if Cost == 'rosenbrock':
    bounds = 3
elif Cost == 'rastrigin':
    bounds = 10
elif Cost == 'pid':
    bounds = 500
# Plotting prepartion
# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# x = np.linspace(-bounds, bounds, 800)
# y = np.linspace(-bounds, bounds, 800)
# X, Y = np.meshgrid(x, y)
# Z = cost([X, Y])


n_particles = 50
dimensions = 3

w = -0.002
w_damp = 0.002
w_min = -2
w_max = -0.1

iter_max = 1

b_low = -bounds
b_upper = bounds

phi_p = 0.2
phi_g = 0.8

particle_position = -np.random.uniform(b_low, b_upper, [dimensions, n_particles])
temp_position = particle_position
best_position = np.array([np.inf, np.inf, np.inf])


for k in range(particle_position.shape[1]):
    if cost(temp_position[:, k]) < cost(best_position):
        best_position = temp_position[:, k]

particle_velocity = np.random.uniform(-np.abs(b_upper - b_low), np.abs(b_upper - b_low), [dimensions, n_particles])


color = iter(plt.cm.plasma(np.linspace(0, 1, 200)))  # after 200 g_best points you lose.
best_scores= [cost(best_position)]
for e in range(iter_max):
    for i in range(n_particles):

        rho_p = np.random.random_sample(1)
        rho_g = np.random.random_sample(1)

        particle_velocity[:, i] = w * particle_velocity[:, i] + phi_p * rho_p * (temp_position[:, i] - particle_position[:, i]) + phi_g * rho_g * (best_position - particle_position[:, i])
        particle_position[:, i] = particle_position[:, i] + particle_velocity[:, i]

        print(particle_position[:, i])
        if cost(particle_position[:, i]) < cost(temp_position[:, i]):
            temp_position[:, i] = particle_position[:, i]

        if cost(particle_position[:, i]) < cost(best_position):
            best_position = particle_position[:, i]
    # ax.cla()
    # ax.plot_wireframe(X, Y, Z, color = 'k', linewidth=0.2)
    # image = ax.scatter3D([
    #     particle_position[0][n] for n in range(n_particles)],
    #     [particle_position[1][n] for n in range(n_particles)],
    #     [cost([particle_position[0][n], particle_position[1][n]]) for n in range(n_particles)], c='r', linewidths=2, marker="^")
    # ax.set(xlim=(-10, 10), ylim=(-10, 10))
    # ax.set_title(["Iteration: ", e, "Minima found at: ", cost([best_position[0], best_position[1]]), " at position: ", best_position])
    # plt.pause(0.05)
    # w = w_min + (w_max - w_min) * (iter_max - k) / iter_max
    w += w * w_damp

print("PSO: ", cost(best_position), best_position)
