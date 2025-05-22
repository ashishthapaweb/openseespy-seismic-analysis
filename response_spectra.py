import numpy as np
from settings import *
import opsvis as opsv

Tn = list(np.arange(0, 6, 0.05))


def soil_type_acc_coeff(analysis_method):
    Sa = []
    if analysis_method == "ESM":
        # IS 1893:2016; for medium stiffness soil
        for T in Tn:
            if 0 < T < 0.55:
                a = 2.5
            elif 0.55 < T < 4.00:
                a = 1.36 / T
            else:
                a = 0.34
            Sa.append(a)
        return [Tn, Sa]

    elif analysis_method == "RSA":
        # IS 1893:2016; for medium stiffness soil
        for T in Tn:
            if T < 0.1:
                a = 1 + 15 * T
            elif 0.1 < T < 0.55:
                a = 2.5
            elif 0.55 < T < 4.00:
                a = 1.36 / T
            else:
                a = 0.34
            Sa.append(a)
        return [Tn, Sa]
