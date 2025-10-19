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


# fig, axes = plt.subplots(1, 2)

# for x in hundred_xs:
# y = eval_2nd_degree(coeffs, x)
# xy_pairs.append((x, y))

# print(xy_pairs)

# xs = []
# ys = []

# for a, b in xy_pairs:
# xs.append(a)
# ys.append(b)

# fig.set_figheight(3)
# fig.set_figwidth(10)

# axes[0].plot(xs, ys, "g+")
# axes[0].set_title("original data")


def eval_2nd_degree_jitter(coeffs, x, j):
    a = coeffs[0] * (x * x)
    b = coeffs[1] * x
    c = coeffs[2]

    y = a + b + c

    interval = [y - j, y + j]
    interval_min = interval[0]
    interval_max = interval[1]

    # print(f"value ranged in {interval_min} and {interval_max}")

    jit_val = random.random() + interval_max

    while interval_min > jit_val:
        jit_val = random.random() + interval_max

    return jit_val


x = 3
j = 4

hundred_xs = np.random.uniform(-10, 10, 100)

xy_pairs = []

for x in hundred_xs:
    y = eval_2nd_degree_jitter(coeffs, x, j)
    xy_pairs.append((x, y))


xs = []
ys = []

for a, b in xy_pairs:
    xs.append(a)
    ys.append(b)

# axes[1].plot(xs, ys, "g+")
# axes[1].set_title("jittered data")
plt.figsize = (10, 20)
plt.plot(xs, ys, "b+")
plt.title("jittered data")

rand_coeffs = (
    random.randrange(-10, 10),
    random.randrange(-10, 10),
    random.randrange(-10, 10),
)

y_bar = eval_2nd_degree(rand_coeffs, hundred_xs)

y_bar_s = []

for n in y_bar:
    y_bar_s.append(n)

# plt.plot(xs, y_bar, "g+", color="black")


def loss_mse(ys, y_bar_s):
    mse = 0

    for i in range(len(hundred_xs)):
        mse += (ys[i] - y_bar_s[i]) ** 2

    mse /= n

    return mse


old_mse = loss_mse(ys, y_bar_s)


def gradient_descent(a_now, b_now, c_now, xs, ys, L):
    a_gradient: float = 0
    b_gradient: float = 0
    c_gradient: float = 0

    n = len(ys)

    for i in range(n):
        x = xs[i]
        y = ys[i]

        a_gradient += -2 * x**2 * (y - (a_now * x**2 + b_now * x + c_now))
        b_gradient += -2 * x * (y - (a_now * x**2 + b_now * x + c_now))
        c_gradient += -2 * (y - (a_now * x**2 + b_now * x + c_now))

    a_gradient /= n
    b_gradient /= n
    c_gradient /= n

    a = a_now - a_gradient * L
    b = a_now - b_gradient * L
    c = a_now - c_gradient * L

    return a, b, c


# a = rand_coeffs[0]
# b = rand_coeffs[1]
# c = rand_coeffs[2]

a: int = 0
b: int = 0
c: int = 0

epochs = 1000
L = 0.0001

for i in range(epochs):
    print(f"epochs: {i}")
    a, b, c = gradient_descent(a, b, c, xs, ys, L)

new_coeffs = [a, b, c]

y_bar = eval_2nd_degree(new_coeffs, hundred_xs)

new_mse = loss_mse(ys, y_bar)

print(f"old mse: {old_mse}\nnew mse: {new_mse}")

plt.plot(
    xs,
    y_bar,
    "g+",
)

plt.title("polynomial regression from scratch")

plt.show()
