import  seaborn as sb
import matplotlib.pyplot as plot

def plot_scoruri(z1, z2, u1, u2, nume_instante):
    fig = plot.figure(figsize=(11, 8))
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plot.Axes)
    ax.scatter(z1, z2, c='r', label="Spatiul 1 (X)")
    ax.scatter(u1, u2, c='b', label="Spatiul 2 (Y)")
    n = len(nume_instante)

    for i in range(n):
        ax.text(z1[i], z2[i], nume_instante[i])
        ax.text(u1[i], u2[i], nume_instante[i])

    ax.legend()
    plot.show()

def corelograma(t, linf = - 1, lsup = 1):
    fig = plot.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    sb.heatmap(t, vmax=lsup, vmin=linf, cmap="bwr", ax=ax)
    plot.show()
