from collections import OrderedDict
from functools import partial
from time import time
import math
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold

import pandas as pd

################################## Data preprocessing ####################################

data = pd.read_csv("../data/dataset.csv")

# Removes empty feature
data.drop(['Unnamed: 83'], axis=1, inplace=True)

# Removes features not usefull
data.drop(['encounter_id', 'patient_id', 'icu_admit_source',
           'icu_id', 'icu_stay_type', 'icu_type'], axis=1, inplace=True)

# Removes rows with at least one null value in all features
data = data[data.isna().sum(axis=1) == 0]

# Correct typo of apache_2_bodysystem "Undefined diagnosis"
data["apache_2_bodysystem"].replace({"Undefined diagnoses": "Undefined Diagnoses"}, inplace=True)

# One hot encoding of string data
for (columnName, columnData) in data.iteritems():
    if columnData.dtype == "object":
        one_hot = pd.get_dummies(columnData, prefix=columnName)
        data.drop([columnName], axis=1, inplace=True)
        data = data.join(one_hot)

print(data.shape)
print(data.nunique())
print(data.info(verbose=True, show_counts=True))
print("No. of rows with missing values:", data.isnull().any(axis=1).sum())

################################### Reduction part ##################################

n_points = 2000
n_neighbors = 15
n_components = 4

total_death = data["hospital_death"]
n_death = len(total_death[total_death == 1])
n_survived = len(total_death) - n_death
death_proportion = n_death / len(total_death)
data_dead = data.loc[(data["hospital_death"] == 1)]
data_survived = data.loc[(data["hospital_death"] == 0)]

X = data_dead.sample(int(n_points * death_proportion))
Y = data_survived.sample(n_points - int(n_points * death_proportion))
sampled_data = pd.concat([X, Y])
Death = sampled_data["hospital_death"]
print(X.info(verbose=True, show_counts=True))
print(Y.info(verbose=True, show_counts=True))
print(sampled_data.info(verbose=True, show_counts=True))
print(death_proportion)

list_comb = list(range(n_components))
list_comb = list(combinations(list_comb, 2))
n_pairs = len(list_comb)

# Create figure
fig, axs = plt.subplots(n_pairs, 2, squeeze=False, figsize=(35, 18))
fig.suptitle(
    "Manifold Learning with %i points, %i neighbors" % (n_points, n_neighbors), fontsize=14
)

# Set-up manifold methods
LLE = partial(
    manifold.LocallyLinearEmbedding,
    n_neighbors=n_neighbors,
    n_components=n_components,
    eigen_solver="dense",
)

methods = OrderedDict()
methods["LLE"] = LLE(method="standard")
methods["Hessian LLE"] = LLE(method="hessian")

# Plot results
for m, (label, method) in enumerate(methods.items()):
    t0 = time()
    Y = method.fit_transform(sampled_data)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    for (l, x) in enumerate(list_comb):
        axs[l, m].scatter(Y[Death == 0, x[0]], Y[Death == 0, x[1]], c="green", label="Survived")
        axs[l, m].scatter(Y[Death == 1, x[0]], Y[Death == 1, x[1]], c="red", label="Died")
        if l == 0:
            axs[l, m].set_title("%s (%.2g sec)" % (label, t1 - t0))
        axs[l, m].xaxis.set_major_formatter(NullFormatter())
        axs[l, m].yaxis.set_major_formatter(NullFormatter())
        axs[l, m].axis("tight")
        axs[l, m].legend()
        axs[l, m].set_xlabel(f"dim : {x[0]}")
        axs[l, m].set_ylabel(f"dim : {x[1]}")

plt.show()
fig.savefig("fig.png")
