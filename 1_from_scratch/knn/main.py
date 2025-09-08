import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from collections import Counter


points = {
    "blue": [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
    "red": [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]],
}

new_point = [4, 3]


def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


def manhattan_distance(p, q):
    return np.sum(np.abs((np.array(p) - np.array(q))))


def minkowski_distance(p, q):
    return (np.sum((abs(np.array(p) - np.array(q))) ** q)) ** (1 / q)


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict_single(self, new_point):
        distances = []
        categories = []

        for category in self.points:
            categories.append(category)
            for point in self.points[category]:  # red, blue
                distance = euclidean_distance(point, new_point)
                distances.append([distance, category])

        sorted_distances = sorted(distances)

        sorted_points = [
            category[1] for category in sorted_distances
        ]  # ['red', 'red', 'red', 'red', 'blue', 'red', 'blue', 'blue', 'blue', 'blue']

        neighbors = sorted_points[: self.k]

        result = Counter(neighbors).most_common(1)[0][0]

        counts = Counter(neighbors)
        print(counts)
        max_count = max(counts.values())
        print(max_count)
        print(counts.items())

        tied = [label for label, c in counts.items() if c == max_count]

        if len(tied) > 1:
            result = Counter(neighbors[:-1]).most_common(1)[0][0]

        return sorted_distances[self.k - 1], result

    def predict(self, new_points):
        results = []
        for p in new_points:
            result = self.predict_single(p)
            results.append(result)

        return results


xy = points["blue"] + points["red"]
# print(xy)

x = []
y = []

colors = 5 * ["blue"] + 5 * ["red"] + ["green"]

for i in xy:
    x.append(i[0])
    y.append(i[1])

x.append(new_point[0])
y.append(new_point[1])


plt.figure(figsize=(8, 8))
plt.scatter(x, y, c=colors)

plt.xlabel("x")
plt.ylabel("y")

clf = KNearestNeighbors(k=4)
clf.fit(points)
limit_point, result = clf.predict_single(new_point)

print(f"result: {result}")

a = new_point[0] + limit_point[0].item() * np.cos(np.linspace(0, 2 * np.pi, 200))
b = new_point[1] + limit_point[0].item() * np.sin(np.linspace(0, 2 * np.pi, 200))

plt.plot(a, b, color="red")

plt.show()
