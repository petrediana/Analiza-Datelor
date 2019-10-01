import pandas as pd
import functii
import numpy as np

fisier = "T:\seminar1\sem1\MortalitateEU\MortalitateEU.csv"
# citire din fisier csv cu specificarea coloanei din care sunt preluate numele de instante
tabel = pd.read_csv(fisier,  index_col=0)
#print(tabel)

#preluare in lista a numelor de coloane
variabile = list(tabel)
variabile_prelucrate = variabile[1:]


# preluare date din tabel in array numpy
x = tabel[variabile_prelucrate].values
#print(x, type(x), sep="\n")

functii.inlocuire_nan(x)

print(functii.acp(x))
