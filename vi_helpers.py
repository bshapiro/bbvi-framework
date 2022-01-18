import autograd.numpy as np
import matplotlib.pyplot as plt

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