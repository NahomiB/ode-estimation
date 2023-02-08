import numpy as np
import matplotlib.pyplot as plt

from modules.edo_model import EDOModel


def scalar_product(f, g, D):

    sum = 0
    for i in range(len(D)):
        sum += f(D[i][0], D[i][1], D[i][2], D[i][3]) * g(D[i][0], D[i][1], D[i][2], D[i][3])
    return sum


def scalar_product_deriv(col, f, D):

    sum = 0
    for i in range(len(D) - 1):
        m = (D[i + 1][col] - D[i][col]) / (D[i + 1][0] - D[i][0])
        sum += f(D[i][0], D[i][1], D[i][2], D[i][3]) * m
    return sum


# Load points data in /data/sir_model.csv
D = np.loadtxt('data/sir_model.csv', delimiter=',', skiprows=1)

f_11 = lambda t, S, I, R : -S * I
f_21 = lambda t, S, I, R : S * I
f_22 = lambda t, S, I, R : - I
f_31 = lambda t, S, I, R : I

system = [[f_11],
          [f_21, f_22],
          [f_31]]

restrictions = [[0, 1], [2, 3]]

sir = EDOModel(system, restrictions, D)
sir.graph(160, 160)

