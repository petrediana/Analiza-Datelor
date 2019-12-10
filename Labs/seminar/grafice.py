import seaborn as sb
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
import sklearn.decomposition as dec

def dendrograma(h, nume_instante, metoda, metrica):
    fig = plt.figure(figsize=(12, 7))
    assert isinstance(fig, plt.Figure)
    ax = fig.subplots()

    assert isinstance(ax, plt.Axes)
    ax.set_title("Grafic dendograma. Metoda: " + metoda + " Metrica: " + metrica, fontsize=16, color='b')
    hclust.dendrogram(h, labels=nume_instante, ax=ax)

    plt.show()

def plot_partitie(x, partitie, nume_instanta):
    fig = plt.figure(figsize=(12, 7))
    assert isinstance(fig, plt.Figure)
    ax = fig.subplots()

    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot partitie", fontsize=16, color='b')

    pca = dec.PCA(n_components=2)
    z = pca.fit_transform(x)

    sb.scatterplot(z[:, 0], z[:, 1], hue=partitie, ax=ax)
    n = len(nume_instanta)
    for i in range (n):
        ax.text(z[i,0], z[i,1], nume_instanta[i])
    plt.show()