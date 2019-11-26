import matplotlib.pyplot as plt
import seaborn as sb

# scoruri axa 1, scoruri axa 2, vector de clasificare, scorurile centrilor pe axa 1, scorurile centrilor pe axa 2 si clasele
def bi_plot(z1, z2, y, zg1, zg2, clase):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    sb.scatterplot(z1, z2, hue=y, ax=ax)

    for clasa in clase:
        sb.scatterplot(zg1, zg2, ax=ax, legend=False, markers="s", s=100)

    plt.show()