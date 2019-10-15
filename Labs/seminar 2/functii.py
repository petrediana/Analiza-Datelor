import numpy as np
import pandas as pd
# definire functie pentru analiza in componente principale
def acp(x):
    n,m = np.shape(x)
    # standardizare x
    x_std = (x-np.mean(x,axis=0))/np.std(x,axis=0)
    r = (1/n)*np.transpose(x_std)@x_std

    #Calcul vectori si valori propri
    valp,vecp = np.linalg.eig(r)
    k = np.flipud(np.argsort(valp))
    alpha = valp[k]
    a = vecp[:, k]

    #calcul componente
    c = x_std @ a

    return r, alpha, a, c


# definire functie pentru inlocuirea valorilor lipsa (nan)
def inlocuire_nan(x):
    # detectie valori lipsa
    is_nan = np.isnan(x)
    k_nan = np.where(is_nan)
    x[k_nan] = np.nanmean(x[:,k_nan[1]],axis=0)


def tabelare_varianta(alpha):
    alpha_cum = np.cumsum(alpha)
    procent_varianta = alpha * 100 / sum(alpha)
    procent_cumulat = np.cumsum(procent_varianta)

    tabelare_varianta = pd.DataFrame(data={
        "Varianta": alpha,
        "Varianta cumulata": alpha_cum,
        "Procent varianta":procent_varianta,
        "Procent cumulat":procent_cumulat
    }, index = ["Comp" + str(i) for i in range(1, len(alpha) + 1)])

    return tabelare_varianta