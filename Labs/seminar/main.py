# ALGORITMI IERARHICI
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hclust
import functii
import grafice

tabel = pd.read_csv("ADN/ADN_Total.csv", index_col=0)
#print(tabel)

nume_instante = list(tabel.index)
#print(nume_instante)

variabile = list(tabel)
#print(variabile)

x = tabel[variabile].values
#print(x)

if np.isnan(x).any():
    print("Valori lipsa!")

# Clasificare ierarhica pe instante
# Construire ierarhie
h = hclust.linkage(x, method="complete") # calculul se bazeaza pe distante | am distanta intre doua instante X, Y
print(h)
#grafice.dendrograma(h, nume_instante, "Legatura completa", "Euclidiana")

# intr o grupare ierarhica pornesc de la: indivic -> cluster (cluster singletone, cu o singura instanta)
# stiu ca sunt (n - 1) jonctiuni

# Identificare partitie de maxima stabilitate
m = np.shape(h)[0]      # 0 imi spune nr de linii
print(m)
k = m - np.argmax(h[1:, 2] - h[:(m-1), 2])     # cate clase are partitia de maxima stabilitate
print(k)

# Determinare partitie
partite = functii.partitie(h, k)
print(partite)

# Tabelare partitie
tabel_partite = pd.DataFrame(data={"PartiiOptimala": partite}, index=nume_instante)
tabel_partite.to_csv("PartitieOptimala.csv")
print(tabel_partite)

# vizualizare clusteri in axele principale
grafice.plot_partitie(x, partite, nume_instante)