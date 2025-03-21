import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# a
A = np.array([
    [1, -2, 3],
    [4, 5, 6],
    [7, 1, 9]
])

# b
b = np.array([1, 2, 3])

# c
x = sp.linalg.solve(A, b)
print(x)

# d
print(np.dot(A, x))

# e
B = np.random.rand(3, 3)
print(B)
print(sp.linalg.solve(A, B))
print(np.dot(A, sp.linalg.solve(A, B)))

# f
eigvals, eigvecs = sp.linalg.eig(A)
print(eigvals, eigvecs)

# g
A_inv = sp.linalg.inv(A)
A_det = sp.linalg.det(A)
print(A_inv)
print(A_det)

# h
print(sp.linalg.norm(A, ord=2))
print(sp.linalg.norm(A, ord=np.inf))