import pandas as pd
import sklearn.cross_decomposition as sdec
import functii
import numpy as np
import grafice
# test -> calcul scoruri plotare || calcul corelatii si tabelarea lor || calcul legaturi (corelatii dintre var observare si var corelate) + plotarea lor [corelograma]

tabel1 = pd.read_csv("MiscareaNaturalaAPop2016/MiscNatAPop_Judete.csv", index_col=0)
tabel2 = pd.read_csv("Teritorial2016/Indicatori_2.csv", index_col=0)

# variabilele prelucrate setul X (tipul fenomen cel al dezv economico-sociala)
var1 = list(tabel1)[1:]
var2 = list(tabel2)[3:]

# instante
tabel = tabel1[var1].join(other=tabel2[var2], how="inner")

# preiau X, Y
x = tabel[var1].values
y = tabel[var2].values
nume_instante = list(tabel.index)

# scap de val null
functii.inlocuire_nan(x)
functii.inlocuire_nan(y)

#print(x, y, sep="\n")

# construire model CCA -> Canonical Correlation Analysis
# calculez nr de radacini canonice
n,p = np.shape(x)
q = y.shape[1]
m = min(p, q) # nr de radacini canonice

# construiesc modelul
cca_model = sdec.CCA(m)
cca_model.fit(x, y)

# preluare rezultate si calcule
# preluare scoruri
z = cca_model.x_scores_
u = cca_model.y_scores_

# cele mai semnificative axe sunt de la 0 -> 1
#grafice.plot_scoruri(z[:, 0], z[:, 1], u[:, 0], u[:, 1], nume_instante)

# calcul corelatii canonice
r = np.diagonal(np.corrcoef(z, u, rowvar=False)[:m, m:]) # sunt asezate pe linii, nu pe coloane; am elementele de pe diagonala principala
print(r) # -> corelatiile

# aplicare test Bartlet-Wilks pentru semnificatie corelatii canonice
# tabelare corelatii canonice

# preluare corelatii dintre variabilele observate si variabilele canonice
# calcul abateri standard pt z si u (variabile canonice)
z_std = np.std(z, axis=0) #var sunt asezate pe coloane
u_std = np.std(u, axis=0)

# corelatii
rxz = cca_model.x_loadings_ * z_std
ryu = cca_model.y_loadings_ * u_std

# trasare corelograme pentru rxz, ryu (sau cerc de corelatie)
t_rxz = pd.DataFrame(data=rxz, index=var1, columns=['z' + str(i) for i in range(1, m + 1)])
t_rxz.to_csv("rxz.csv")

grafice.corelograma(t_rxz)
