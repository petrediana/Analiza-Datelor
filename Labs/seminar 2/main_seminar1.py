import pandas as pd
import numpy as np
import functii
import grafice

fisier = "MortalitateEU/MortalitateEU.csv"
# citire din fisier csv cu specificarea coloanei din care sunt preluate
# numele de instante

tabel = pd.read_csv(fisier,index_col=0)
# print(tabel)
# preluare in lista a numelor de coloane
variabile = list(tabel)
# print(variabile)

variabile_prelucrate = variabile[1:]
instante = list(tabel.index)

# print(variabile_prelucrate)
# Preluare date din tabel in array numpy
x = tabel[variabile_prelucrate].values
# print(x,type(x),sep="\n")

functii.inlocuire_nan(x)
r, alpha, a, c = functii.acp(x)
#print(r)

#Calcul componente, tabelare varianta
tabel_varianta = functii.tabelare_varianta(alpha)
print(tabel_varianta)
tabel_varianta.to_csv("varianta.csv")

#Grafic varianta
    #grafice.plot_varianta(alpha)

#calcul scoruri
s = c / np.sqrt(alpha)

#plot scoruri
kx = 0
ky = 1
#grafice.plot_scoruri(s[:, kx], s[:, ky], kx, ky, instante)

#calcul corelatii dintre variabile observate si componente
rxc = a * np.sqrt(alpha)

#corelograma -> grafic specializat
t_rxc = pd.DataFrame(data = rxc, index = variabile_prelucrate, columns = ["comp1" + str(i) for i in range(1, len(alpha) + 1)])
t_rxc.to_csv("rxc.csv")
grafice.corelograma(t_rxc)