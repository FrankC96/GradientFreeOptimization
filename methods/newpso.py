import numpy as np
import matplotlib.pyplot as plt

from cost_function import cost
# x : position
# v :  velocity
# p_best : iteration's best
# g_best : global' s best

bounds = 10
b_upper = bounds
b_low = -bounds
c1 = 0.2
c2 = 0.8
w = 0.5

x = np.random.uniform(b_low, b_upper, [2, 20])  # no. dimenions, no. particles
v = np.zeros([2, 20])

v_max = 300
v_min = -300

p_best = x
g_best = np.Inf * np.ones(2)
for k in range(1000):  # maximum iterations
    for t in range(20-1):  # pop size

        if cost(x[:, t]) < cost(p_best[:, t]):
            p_best[:, t] = x[:, t]
            if cost(g_best) < cost(p_best):
                g_best = x[:, t]

        for d in range(2):  # no. of dimensions
            r1 = np.random.random_sample(1)
            r2 = np.random.random_sample(1)

            v[:, t+1] = w * v[:, t] + c1 * r1 * (p_best[:, t] - x[:, t]) + c2 * r2 * (g_best - x[:, t])
            x[:, t+1] = x[:, t] + v[:, t]

            if v[:, t+1].any() > v_max:
                v[:, t+1] = v_max
            elif v[:, t+1].any() < v_min:
                v[:, t+1] = v_min

            if x[:, t+1].any() > v_max:
                x[:, t+1] = v_max
            elif x[:, t+1].any() < v_min:
                x[:, t+1] = v_min
        print(g_best)
        plt.plot(k, cost(g_best), 'x')
        plt.pause(0.005)