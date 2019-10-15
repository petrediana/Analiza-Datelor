import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

#plot varianta
def plot_varianta(alpha):
    m = len(alpha) #numar componente

    fig = plt.figure(figsize=(10, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)

    assert isinstance(ax,plt.Axes)
    ax.set_title("Plot varianta")
    ax.set_xlabel("Componenta")
    ax.set_ylabel("Varianta")

    x = np.array([i for i in range(1, m + 1)])
    ax.plot(x, alpha, 'b')
    ax.scatter(x, alpha, c = 'r')
    ax.set_xticks(x)
    ax.axhline(1, c = 'g')

    e = alpha[:(m - 1)] - alpha[1:]
    sigma = e[: (m - 2)] - e[1:]

    res = sigma < 0
    if any(res):
        k = np.where(sigma < 0)
        ax.axhline(alpha[k[0][0] + 1])

    plt.show()


#plot scoruri
def plot_scoruri(x, y, kx = 0, ky = 1, etichete = None):
    n = len(etichete)

    fig = plt.figure(figsize=(10, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)

    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot scoruri")
    ax.set_xlabel("a" + str(kx + 1))
    ax.set_ylabel("a" + str(ky + 1))

    ax.scatter(x, y, c = 'r')
    ax.axhline(0, c = 'g')
    ax.axvline(0, c = 'g')


    if etichete is not None:
        for i in range(0, n):
            ax.text(x[i], y[i], etichete[i])

    plt.show()


#corelograma
def corelograma(t, vmin = -1, vmax = 1):
    sb.heatmap(t, vmin = vmin, vmax = vmax, cmap='RdYlBu')
    plt.show()
