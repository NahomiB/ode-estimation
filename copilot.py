import numpy as np

# Solve a linear equation sysytem of m equations in n unknowns using QR factorization
# Input: A is an m x n matrix, b is an m x 1 vector
# Output: x is an n x 1 vector
def solve(A, b):
    Q, R = np.linalg.qr(A)
    y = np.dot(Q.T, b)
    x = np.linalg.solve(R, y)
    return x

# A method to solve a non-square system of linear equations using QR factorization
# Input: A is an m x n matrix, b is an m x 1 vector
# Output: x is an n x 1 vector
def solve_non_square(A, b):
    Q, R = np.linalg.qr(A)
    y = np.dot(Q.T, b)
    x = np.linalg.lstsq(R, y, rcond=None)[0]
    return x

# Implement tests here
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
b = np.array([[3], [3], [4]])
x = solve_non_square(A, b)
print(x)

# Implement a test with m > n
A = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([[3], [3]])
x = solve_non_square(A, b)
print(x)

# Implement a test with 10 equations and 3 unknowns
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [11, 12, 13], [14, 15, 16], [17, 18, 19], [20, 21, 22], [23, 24, 25], [26, 27, 28], [29, 30, 31]])
b = np.array([[3], [3], [4], [5], [6], [7], [8], [9], [10], [11]])
x = solve_non_square(A, b)
print(x)

# Implement a test with 5 equations and 3 unknowns with an inconsistent system
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [11, 12, 13], [14, 15, 16]])
b = np.array([[3], [3], [4], [5], [6]])
x = solve_non_square(A, b)
print(x)

# Implement a test with 4 equations and 3 unknowns with an inconsistent system
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10], [11, 12, 13]])
b = np.array([[3], [3], [4], [5]])
x = solve_non_square(A, b)
print(x)
