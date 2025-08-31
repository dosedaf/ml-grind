import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

coeffs = [2, -5, 4]


def eval_2nd_degree(coeffs, x):
    a = coeffs[0] * (x * x)
    b = coeffs[1] * x
    c = coeffs[2]

    y = a + b + c

    return y


hundred_xs = np.random.uniform(-10, 10, 100)
print(hundred_xs)

xy_pairs = []

for x in hundred_xs:
    y = eval_2nd_degree(coeffs, x)
    xy_pairs.append((x, y))

print(xy_pairs)

xs = []
ys = []

for a, b in xy_pairs:
    xs.append(a)
    ys.append(b)

plt.figure(figsize=(20, 10))
plt.plot(xs, ys, "g+")
plt.title("original data")
plt.show()


def eval_2nd_degree_jitter(coeffs, x, j):
    a = coeffs[0] * (x * x)
    b = coeffs[1] * x
    c = coeffs[2]

    y = a + b + c

    interval = [y - j, y + j]
    interval_min = interval[0]
    interval_max = interval[1]

    print(f"value ranged in {interval_min} and {interval_max}")

    jit_val = random.random() + interval_max

    while interval_min > jit_val:
        jit_val = random.random() + interval_max

    return jit_val


x = 3
j = 4

print(eval_2nd_degree_jitter(coeffs, x, j))
