from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_3d(points, points_color, title):
    x, y, z = points.T
    fig, ax = plt.subplots(figsize=(4, 4), facecolor="white", subplot_kw={"projection": "3d"})
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)
    ax.set_zticks([])
    ax.set_zticks([], minor=True)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    # fig.subplots_adjust(top=0.2, bottom=0, left=0, right=1)
    fig.savefig("./plots/swiss_roll_3d.svg", bbox_inches='tight', pad_inches=0)




def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_xticks([])
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    ax.set_yticks([], minor=True)


sh_points, sh_color = datasets.make_swiss_roll(n_samples=1500, random_state=617)
plot_3d(sh_points, sh_color, "Original Swiss Hole")
n_neighbors = 12  # neighborhood which is used to recover the locally linear structure
n_components = 2  # number of coordinates for the manifold
params = {"n_neighbors": n_neighbors,
          "n_components": n_components,
          "eigen_solver": "auto",
          "random_state": 617}
lle_standard = manifold.LocallyLinearEmbedding(method="standard", **params)
sh_standard = lle_standard.fit_transform(sh_points)
lle_mod = manifold.LocallyLinearEmbedding(method="modified", modified_tol=0.8, **params)
sh_mod = lle_mod.fit_transform(sh_points)
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), facecolor="white")
name, points = ("LLE", sh_standard)
add_2d_scatter(ax, points, sh_color, name)
fig.tight_layout()
fig.savefig("./plots/swiss_roll_lle.svg")
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3), facecolor="white")
name, points = ("MLLE", sh_mod)
add_2d_scatter(ax, points, sh_color, name)
fig.tight_layout()
fig.savefig("./plots/swiss_roll_mlle.svg")