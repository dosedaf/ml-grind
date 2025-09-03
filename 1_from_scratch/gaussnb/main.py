import numpy as np
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
    def fit(self, X, y):
        a = {}
        classes = np.unique(y)

        for c in classes:
            a[c] = []

        for c in classes:  # loop over classes
            for i in enumerate(y):  # loop over the enumeration of y, {1, female}
                if i[1] == c:  # if i[1] (female) is the same as c, looped class
                    a[c].append(X[i[0]])  # if it is, then append that to the col

        print(a["male"])


gauss = GaussianNB()

X = df["Height"].values
y = df["Gender"].values

gauss.fit(X, y)
