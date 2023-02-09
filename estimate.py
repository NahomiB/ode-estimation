import numpy as np
import matplotlib.pyplot as plt

from modules.edo_model import EDOModel

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
print(sir.params)
sir.graph(0, 100, 100)