import numpy as np
import pandas as pd

# trimit matricea ierarhie si cate clase vreau sa trimit
def partitie(h, k):
    n = np.shape(h)[0] + 1 # numarul de instante
    c = np.arange(n) # primii n clusteri

    for i in range(n - k):
        k1 = h[i, 0]
        k2 = h[i, 1]

        # se formeaza cluster n + i si trebuie sa includa toate instantele care erau in k1 si k2
        c[c==k1] = n + i
        c[c==k2] = n + i

    #print(c)
    c_transformat_categorie = pd.Categorical(c).codes # imi trasforma, imi intoarce in c_trans variabila categoriala cu cele k categorii
    return ["c" + str(i + 1) for i in c_transformat_categorie]