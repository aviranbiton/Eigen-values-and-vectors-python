import time
from math import sqrt, inf
import numpy as np


def save_matrix(name, a):
    np.savetxt(name, a, "%.3f")
    print("Done Saving Matrix to file.")


def QR_Decomposition(A):
    n, m = A.shape  # get the shape of A

    Q = np.empty((n, n))  # initialize matrix Q
    u = np.empty((n, n))  # initialize matrix u

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] * (1 / np.linalg.norm(u[:, 0]))

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]  # get each u vector

        Q[:, i] = u[:, i] / norm(u[:, i])  # compute each e vetor

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R


def eigval2(X):
    n, m = X.shape
    q, r = QR_Decomposition(X)
    i = 0
    diff = inf
    old = q
    while diff > 1e-12:
        X = r @ q
        q1, r1 = QR_Decomposition(X)
        diff = norm(np.diag(r) - np.diag(r1))
        old = old @ q1
        q = q1
        r = r1
        i = i + 1

    r = np.diag(r * q) * np.identity(n)
    q = old
    return q, r


def norm(V):
    res = 0
    for i in range(V.shape[0]):
        res += V[i] * V[i]
    return sqrt(res)


# main

matrix = np.loadtxt("eig.txt", delimiter=",")
print("Matrix Loaded. Size:", matrix.shape)
start = time.time()
a, b = eigval2(matrix)
print(time.time() - start)

# saving results

save_matrix("vectors.txt", a)
save_matrix("values.text", b)

# checking results

c = matrix @ a
d = a @ b
