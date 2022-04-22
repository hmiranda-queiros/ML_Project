from collections import OrderedDict
from functools import partial
from time import time

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

nb_points = 2000
X = data[:nb_points]
Death = X["hospital_death"]
n_neighbors = 10
n_components = 2

# Create figure
fig = plt.figure(figsize=(15, 8))
fig.suptitle(
    "Manifold Learning with %i points, %i neighbors" % (nb_points, n_neighbors), fontsize=14
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
for i, (label, method) in enumerate(methods.items()):
    t0 = time()
    Y = method.fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(1, 2, 1 + i)
    ax.scatter(Y[Death == 0, 0], Y[Death == 0, 1], c="green", label="Survived")
    ax.scatter(Y[Death == 1, 0], Y[Death == 1, 1], c="red", label="Died")
    ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")
    ax.legend()

plt.show()
