import matplotlib.pyplot as plt


def mimshow(im, title=None):
    fig, ax = plt.subplots()
    imm = ax.imshow(im)
    fig.colorbar(imm, ax=ax)
    if title:
        ax.set_title(title)
    fig.show()
    return fig, ax


def mplot(X, Y=None, title=None):
    fig, ax = plt.subplots()
    if Y:
        ax.plot(X, Y)
    else:
        ax.plot(X)

    if title:
        ax.set_title(title)
    fig.show()
    return fig, ax

