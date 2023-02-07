import numpy as np
import matplotlib.pyplot as plt

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

normal_mat = np.zeros((6, 6))

current_row = 0
for i in range(3): # For each equation...
    for j in range(len(system[i])): # ...and for each parameter...
        for k in range(j, len(system[i])): # ...and for each parameter again...

            normal_mat[current_row + j][current_row + k] = scalar_product(system[i][j], system[i][k], D)
            normal_mat[current_row + k][current_row + j] = normal_mat[current_row + j][current_row + k]
    current_row += len(system[i])

normal_mat[4][0] = 1
normal_mat[4][1] = -1
normal_mat[0][4] = 1
normal_mat[1][4] = -1

normal_mat[5][2] = 1
normal_mat[5][3] = -1
normal_mat[2][5] = 1
normal_mat[3][5] = -1

print(normal_mat)

# Vector of size 6 full of 0
v = np.zeros(6)

cont = 0
for eq in range(len(system)):
    for param in range(len(system[eq])):
        v[cont] += scalar_product_deriv(eq + 1, system[eq][param], D)
        cont += 1

# Solve the system
x = np.linalg.solve(normal_mat, v)
print(x)
