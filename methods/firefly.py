import numpy as np
import matplotlib.pyplot as plt

from cost_function import cost
low_pos = -10
upp_pos = 10
dimensions = 2

# Initialize population of m fireflies
x = np.random.uniform(low_pos, upp_pos, [dimensions, 100]) # no. of fireflies

# light intensity
I = np.zeros(x.shape[1])
for i in range(x.shape[1]):
    I[i] = cost(x[:, i])

I0 = 0
beta0 = 2
alpha = 0.2
I_i = np.zeros(x.shape[1])

for _ in range(1000):
    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            epsilon = np.random.random_sample()
            gamma = 1
            r = np.linalg.norm(x[:, i] - x[:, j], ord=2)
            beta = beta0 * np.exp(-gamma * r**2)

            I_i[i] = I0 * np.exp(-gamma * r)
            if cost(np.array(x[:, i])) > cost(np.array(x[:, j])):
                #print(cost(x[:, i]))
                x[:, i] = x[:, i] + beta * (x[:, j] - x[:, i]) + alpha * epsilon

    for k in range(x.shape[1]):
        I[k] = cost(x[:, k])
    #print("Ordered intensities: ", I)
    print(x[:, I.argmin()], "with cost: ", cost(x[:, I.argmin()]))