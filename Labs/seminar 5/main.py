import numpy as np
import pandas as pd
import sklearn.discriminant_analysis as disc
import sklearn.metrics as metrics
import grafice
import sklearn.naive_bayes as nb

tabela = pd.read_csv("Hernia/hernia.csv", index_col=0)
variabile = list(tabela)
nr_variabile = len(variabile)
variabile_predictor = variabile[:(nr_variabile - 1)] # toate mai putin ultima
variabile_tinta = variabile[nr_variabile - 1]

# tabelul de date aferent predictorilor
x = tabela[variabile_predictor].values
y = tabela[variabile_tinta].values

# print(x) print(y)

# aplicam modelul + construire model
model_analiza_liniara_discriminanta = disc.LinearDiscriminantAnalysis()
model_analiza_liniara_discriminanta.fit(x, y) # trimit input

# preluare rezultate si aplicare model
clase = model_analiza_liniara_discriminanta.classes_

# calcul scoruri discriminante
z = model_analiza_liniara_discriminanta.transform(x)

# calcul scoruri centrii grupelor
g = model_analiza_liniara_discriminanta.means_ # centrii
zg = model_analiza_liniara_discriminanta.transform(g) # calculeaza scorurile g -> determina coef axiali (g1, g2... sunt centrii)

 # grafice.bi_plot(z[:, 0], z[:, 1], y, zg[:, 0], zg[:, 1], clase)

# clasificare in setul de antrenament
clasificare = model_analiza_liniara_discriminanta.predict(x)
instante = list(tabela.index)
tabel_clasificare = pd.DataFrame(data={"Clasa": y, "Predictie": clasificare}, index=instante)
tabel_clasificare.to_csv("ClasificareBaza.csv")

tabel_mal_clasificare = tabel_clasificare[y != clasificare]
tabel_mal_clasificare.to_csv("ClasificariEronate.csv")

# calcul matrice mal_clasificari (matrice de confuzie)
matrice_confuzie = metrics.confusion_matrix(y, clasificare)
# print(matrice_confuzie)
t_matrice_confuzie = pd.DataFrame(matrice_confuzie, index=clase, columns=clase) # o transform mai frumos
t_matrice_confuzie["Acuratete"] = np.diagonal(matrice_confuzie) * 100 / np.sum(matrice_confuzie, axis=1)

print(t_matrice_confuzie) # in linii real, in coloane estimat
# 31 de instante din hernie au fost clasificate in clasa hernia; 31 din 46 bine clasificate

acuratete_globala = metrics.accuracy_score(y, clasificare)
print("Acuratetea globala:", acuratete_globala * 100, "%")


# aplicam modelul pentru predictia setului de test (hernia_test.csv, aici nu am ultima coloana de predictie)
tabela_test = pd.read_csv("Hernia/hernia_test.csv", index_col=0)
x_test = tabela_test[variabile_predictor].values
clasificare_test = model_analiza_liniara_discriminanta.predict(x_test)
tabela_test["Predictie"] = clasificare_test

tabela_test.to_csv("ClasificareHerniaTest.csv")

model_B = nb.GaussianNB()
model_B.fit(x, y)

clasificare_model_B = model_B.predict(x)
acuratete_globala_B = metrics.accuracy_score(y, clasificare_model_B)
print("Acuratete globala_B", acuratete_globala_B * 100, "%")