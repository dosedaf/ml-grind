import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

data = pd.read_csv("data.csv")

plt.scatter(data.SAT, data.GPA)


def loss_function(m, b, points):
    total_error = 0

    for i in range(len(points)):
        x = points.iloc[i].SAT
        y = points.iloc[i].GPA
        total_error += (y - (m * x + b)) ** 2
    total_error / float(len(points))


def gradient_descent(m_now, b_now, points, L):
    m_gradient: float = 0
    b_gradient: float = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].SAT
        y = points.iloc[i].GPA

        m_gradient += -2 * x * (y - (m_now * x + b_now))
        b_gradient += -2 * (y - (m_now * x + b_now))

        # m_gradient = x * (y - (m_now * x + b_now))
        # b_gradient = y - (m_now * x - b_now)

    m_gradient /= n
    b_gradient /= n

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L

    return m, b


m: int = 0
b: int = 0
L = 0.00000001
epochs: int = 500

print(f"type of m: {type(m)}")

for i in range(epochs):
    print(f"epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print(m, b)

print(f"min : {data.SAT.min()}")
print(f"max : {data.SAT.max()}")

print(f"min : {data.GPA.min()}")
print(f"max : {data.GPA.max()}")

plt.scatter(data.SAT, data.GPA, color="black")
plt.plot(list(range(1630, 2055)), [m * x + b for x in range(1630, 2055)], color="red")
plt.show()
