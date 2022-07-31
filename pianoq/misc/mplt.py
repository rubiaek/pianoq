import matplotlib.pyplot as plt


def mimshow(im, title=None, aspect=None, **args):
    fig, ax = plt.subplots()
    imm = ax.imshow(im, aspect=aspect, **args)
    fig.colorbar(imm, ax=ax)
    if title:
        ax.set_title(title)
    fig.show()
    return fig, ax


def mplot(X, Y=None, title=None):
    fig, ax = plt.subplots()
    if Y is not None:
        ax.plot(X, Y)
    else:
        ax.plot(X)

    if title:
        ax.set_title(title)
    fig.show()
    return fig, ax

