import autograd.numpy as np
import matplotlib.pyplot as plt
import autograd.scipy.stats.multivariate_normal as mvn
import pandas as pd
from numpy.random import multivariate_normal


def all_same(items):
    return all(np.array_equal(x, items[0]) for x in items)


def size(item):
    try:
        return max(item.shape[0], item.shape[1])
    except:
        try:
            return len(item)
        except:
            return 1


def softplus(value):
    if size(value) > 1:
        # import pdb; pdb.set_trace()
        return np.array([softplus(item) for item in value])
    else:
        return threshold(value)


def hardplus(value):
    if value <= 0:
        return 0
    else:
        return value


def regularize(item, diagonal=False):
    if diagonal:
        return item + np.identity(item.shape[0]) * 0.000001
    else:
        return item + 0.000001


def threshold(item):
    if item < 600:
        return np.log(np.exp(item) + 1)
    else:
        return item


def plot_isocontours(ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
    x = np.linspace(*xlimits, num=numticks)
    y = np.linspace(*ylimits, num=numticks)
    X, Y = np.meshgrid(x, y)
    zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
    Z = zs.reshape(X.shape)
    plt.contour(X, Y, Z)
    ax.set_yticks([])
    ax.set_xticks([])


def generate_clusters(n, sigma):
    clusters = [[-2, 0], [2, 0], [-2, -4], [2, -4]]
    ns = [n, n, n, n]
    sigmas = [sigma, sigma, sigma, sigma]
    samples = []
    for cluster in range(len(clusters)):
        samples.extend(multivariate_normal(clusters[cluster], sigmas[cluster] * np.eye(len(clusters[0])), size=ns[cluster]))
    return np.array(samples)


def plot_clusters(clusters, samples, labels, title):
    sample_x = [sample[0] for sample in samples]
    sample_y = [sample[1] for sample in samples]
    labels = labels
    df = pd.DataFrame({"x": sample_x, "y": sample_y, "cluster": labels})
    groups = df.groupby("cluster")
    plt.title(title)
    for name, group in groups:
        plt.scatter(group["x"], group["y"], label=name)
    if clusters is not None:
        plt.scatter([cluster[0] for cluster in clusters], [cluster[1] for cluster in clusters], s=50, c='black')


def visualize_clusters(ax, samples, labels, means, covs):
    plot_clusters(None, samples, labels, '')
    for ci in range(len(means)):
        variational_contour = lambda x: mvn.pdf(x, means[ci], covs[ci])
        plot_isocontours(ax, variational_contour, xlimits=ax.get_xlim(), ylimits=ax.get_ylim())
    plt.legend()
