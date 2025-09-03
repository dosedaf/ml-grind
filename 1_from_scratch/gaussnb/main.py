import numpy as np
import math
import pandas as pd


def generate_dummy_dataset():
    np.random.seed(42)

    male_heights = np.random.normal(loc=175, scale=7, size=50)
    female_heights = np.random.normal(loc=162, scale=6, size=50)

    heights = np.concatenate([male_heights, female_heights])
    genders = np.array(["male"] * 50 + ["female"] * 50)

    df = pd.DataFrame({"Height": heights, "Gender": genders})

    return df


df = generate_dummy_dataset()


class GaussianNB:
    data = {}
    means = {}
    std = {}
    classes = []
    gaussian_prob: float = 0

    def fit(self, X, y):
        self.classes = np.unique(y)

        for c in self.classes:
            self.data[c] = []

        for c in self.classes:  # loop over classes
            for i in enumerate(y):  # loop over the enumeration of y, {1, female}
                if i[1] == c:  # if i[1] (female) is the same as c, looped class
                    self.data[c].append(
                        X[i[0]]
                    )  # if it is, then append that to the col

        # print(f"male : {a['female']}\n")
        # print(f"female : {a['female']}\n")

        # 1 / nk * i: y = k sigma Xij, j = feature

        self.get_mean()

        self.get_standard_deviation()
        self.priors = {}
        for c in self.classes:
            self.priors[c] = len(self.data[c]) / len(self.classes)
            print(self.means, self.std)

    def get_mean(self):
        means = {}
        total = 0

        for c in self.classes:
            # means[c] = np.mean(data[c]) lol
            total = 0

            for x in self.data[c]:
                total += x

            # print(f"len {c}: {len(a[c])}")
            # print(f"total {c}: {total}")

            means[c] = total / len(self.data[c])

        self.means = means

    def get_standard_deviation(self):
        std = {}

        for c in self.classes:
            total = 0

            for x in self.data[c]:
                total += (x - self.means[c]) ** 2

            std[c] = math.sqrt(total / len(self.data[c]))

        self.std = std

    def gaussian_prob(self, x, mean, std):
        print(x, mean, std)
        exponent = math.exp(-((x - mean) ** 2 / (2 * std**2)))
        print(exponent)

        p = exponent / math.sqrt(2 * math.pi * std**2)

        gaussian_prob = p

        return gaussian_prob

    def log_gaussian_prob(self, x, mean, std, eps=1e-9):
        var = std**2 + eps
        return -0.5 * (math.log(2 * math.pi * var) + ((x - mean) ** 2) / var)

    def predict_single(self, x):
        scores = {}
        for c in self.classes:
            log_prior = math.log(self.priors[c])
            log_likelihood = self.log_gaussian_prob(x, self.means[c], self.std[c])
            scores[c] = log_prior + log_likelihood
        return max(scores, key=scores.get)  # class with highest score

    def predict(self, X):
        return [self.predict_single(x) for x in X]


# def predict_singel(self, x):


gauss = GaussianNB()

X = df["Height"].values
y = df["Gender"].values

gauss.fit(X, y)

p_male = gauss.gaussian_prob(180, gauss.means["male"], gauss.std["male"])
p_female = gauss.gaussian_prob(180, gauss.means["female"], gauss.std["female"])

print(f"p male : {p_male} \np female : {p_female}")

gauss = GaussianNB()
X = df["Height"].values
y = df["Gender"].values
gauss.fit(X, y)

print("Single test:")
print("Predict 180cm:", gauss.predict_single(180))
print("Predict 160cm:", gauss.predict_single(160))

print("Batch test:")
preds = gauss.predict([170, 165, 180])
print(preds)
