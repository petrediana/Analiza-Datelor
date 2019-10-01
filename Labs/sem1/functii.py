import numpy as np

#definire functie pentru analiza in componente principale
def acp(x):
    n,m = np.shape(x)

    #standardizare x
    x_std = (x - np.mean(x, axis=0)) / np.std(x, axis=0) #calcul de medie pe coloana
    r = (1 / n) * np.transpose(x_std) @ x_std

    return r

#definire functie pentru inlocuirea valorilor lipsa (nan)
def inlocuire_nan(x):
    is_nan = np.isnan(x) # detectie valori lipsa, is_nan este o matrice cu valori bool
    k_nan = np.where(is_nan)

    x[k_nan] = np.nanmean(x[:,k_nan[1]], axis=0) #calculez media ignorand valorile lipsa

