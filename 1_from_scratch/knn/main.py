import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


points = {
    "blue": [[2, 4], [1, 3], [2, 3], [3, 2], [2, 1]],
    "red": [[5, 6], [4, 5], [4, 6], [6, 6], [5, 4]],
}

new_point = [4, 3]
new_points = [[4, 3], [3, 5], [7, 5]]


def tiebreak(neighbors, tie_num):
    seen = {}
    for neighbor in neighbors:
        seen[neighbor] = 0

    for neighbor in neighbors:
        seen[neighbor] += 1
        if seen[neighbor] == tie_num:
            return neighbor

    return None


def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))


def manhattan_distance(p, q):
    return np.sum(np.abs((np.array(p) - np.array(q))))


def minkowski_distance(p, q, order=3):
    return (np.sum((abs(np.array(p) - np.array(q))) ** order)) ** (1 / order)


class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k
        self.points = None

    def fit(self, points):
        self.points = points

    def predict_single(self, new_point, distance_fn=euclidean_distance):
        distances = []
        categories = []

        for category in self.points:
            categories.append(category)
            for point in self.points[category]:  # red, blue
                distance = distance_fn(point, new_point)
                distances.append([distance, category])

        sorted_distances = sorted(distances, key=lambda x: x[0])

        sorted_points = [
            category[1] for category in sorted_distances
        ]  # ['red', 'red', 'red', 'red', 'blue', 'red', 'blue', 'blue', 'blue', 'blue']

        neighbors = sorted_points[: self.k]

        result = Counter(neighbors).most_common(1)[0][0]

        counts = Counter(neighbors)

        max_count = max(counts.values())

        tied = [label for label, c in counts.items() if c == max_count]

        # print(Counter(neighbors).most_common(1)[0][1])

        if len(tied) > 1:
            result = tiebreak(neighbors, Counter(neighbors).most_common(1)[0][1])

        return sorted_distances[self.k - 1], result

    def predict(self, new_points, distance_fn=euclidean_distance):
        results = []
        kth_neighbors = []
        for p in new_points:
            kth_neighbor, result = self.predict_single(p, distance_fn=distance_fn)
            results.append(result)
            kth_neighbors.append(kth_neighbor)

        return kth_neighbors, results


def plotable(points, new_points):
    xy = points["blue"] + points["red"]

    x = []
    y = []

    colors = 5 * ["blue"] + 5 * ["red"] + len(new_points) * ["green"]

    for i in xy:
        x.append(i[0])
        y.append(i[1])

    for new_point in new_points:
        x.append(new_point[0])
        y.append(new_point[1])

    return x, y, colors


x, y, colors = plotable(points, new_points)

plt.figure(figsize=(15, 15))
plt.scatter(x, y, c=colors)

plt.xlabel("x")
plt.ylabel("y")

clf = KNearestNeighbors(k=4)
clf.fit(points)

kth_neighbor, result = clf.predict_single(new_point, distance_fn=minkowski_distance)

radius = kth_neighbor[0]

kth_neighbors, results = clf.predict(new_points, distance_fn=manhattan_distance)

print(f"result: {result}")
print(f"results: {results}")


def plot_circle(new_points, kth_neighbors):
    print(kth_neighbors[1])
    for i in range(len(new_points)):
        a = new_points[i][0] + kth_neighbors[i][0] * np.cos(
            np.linspace(0, 2 * np.pi, 200)
        )
        b = new_points[i][1] + kth_neighbors[i][0] * np.sin(
            np.linspace(0, 2 * np.pi, 200)
        )
        plt.plot(a, b, color="red")


plot_circle(new_points, kth_neighbors)
plt.show()
