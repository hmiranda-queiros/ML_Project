from sklearn import manifold, datasets
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import ticker
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def plot_3d(points, points_color, title):
    x, y, z = points.T
    fig, ax = plt.subplots(
        figsize=(6, 6),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=50, alpha=0.8)
    ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
    fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    plt.show()


def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())

sh_points, sh_color = datasets.make_swiss_roll(n_samples=1500, hole=True, random_state=617)
# plot_3d(sh_points, sh_color, "Original Swiss Hole")

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

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(7, 7), facecolor="white", constrained_layout=True)
#fig.suptitle("Locally Linear Embeddings", size=16)
methods = [("Original Swiss hole", sh_points),
           ("Standard locally linear embedding", sh_standard),
           ("Modified locally linear embedding", sh_mod)]
for ax, method in zip(axs.flat, methods):
    name, points = method
    add_2d_scatter(ax, points, sh_color, name)
plt.show()
fig.savefig("./plots/swiss_hole.svg")

#
#
#
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d")
# fig.add_axes(ax)
# ax.scatter(sh_points[:, 0], sh_points[:, 1], sh_points[:, 2], c=sh_color, s=50, alpha=0.8)
# ax.set_title("Swiss-Hole in Ambient Space")
# ax.view_init(azim=-66, elev=12)
# _ = ax.text2D(0.8, 0.05, s="n_samples=1500", transform=ax.transAxes)
#
#
#
#
#
#
# LLE = manifold.LocallyLinearEmbedding()
#               eigen_solver='auto',
#               neighbors_algorithm='auto',
#               random_state=617)
# methods['LLE'] = LLE(n_components=12, n_neighbors=14, method="standard")
#
# sh_lle, sh_err = manifold.locally_linear_embedding(
#     sh_points, n_neighbors=12, n_components=2
# )
#
# sh_tsne = manifold.TSNE(
#     n_components=2, learning_rate="auto", perplexity=40, init="random", random_state=0
# ).fit_transform(sh_points)
#
# fig, axs = plt.subplots(figsize=(8, 8), nrows=2)
# axs[0].scatter(sh_lle[:, 0], sh_lle[:, 1], c=sh_color)
# axs[0].set_title("LLE Embedding of Swiss-Hole")
# axs[1].scatter(sh_tsne[:, 0], sh_tsne[:, 1], c=sh_color)
# _ = axs[1].set_title("t-SNE Embedding of Swiss-Hole")