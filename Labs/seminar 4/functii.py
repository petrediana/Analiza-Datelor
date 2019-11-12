import numpy as np

def inlocuire_nan(x):
    k = np.where(np.isnan(x))
    x[k] = np.nanmean(x[:, k[1]], axis = 0)