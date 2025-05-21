import numpy as np
import openseespy.opensees as ops


def CQC(mu, lambdas, dmp, scalf):
    u = 0.0
    ne = len(lambdas)
    for i in range(ne):
        for j in range(ne):
            di = dmp[i]
            dj = dmp[j]
            bij = lambdas[i] / lambdas[j]
            rho = ((8.0 * np.sqrt(di * dj) * (di + bij * dj) * (bij ** (3.0 / 2.0))) /
                   ((1.0 - bij ** 2.0) ** 2.0 + 4.0 * di * dj * bij * (1.0 + bij ** 2.0) +
                    4.0 * (di ** 2.0 + dj ** 2.0) * bij ** 2.0))
            u += scalf[i] * mu[i] * scalf[j] * mu[j] * rho
    return np.sqrt(u)


def rsm_analysis(Tn, Sa):
    ops.timeSeries("Path", 2, "-time", *Tn, "-values", *Sa)
    tsTag = 2
    direction = 1  # excited DOF = Ux
    # Damping and scale factor
    # dmp = [0.05] * len(eigenValues)
    # scalf = [1.0] * len(eigenValues)
