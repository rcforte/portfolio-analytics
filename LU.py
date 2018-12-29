import scipy.linalg as linalg
import numpy as np

A = np.array([[2., 1., 1.],
              [1., 3., 2.],
              [1., 0., 0.]])
B = np.array([4., 5., 6.])

LU = linalg.lu_factor(A)
x = linalg.lu_solve(LU,B)

print(x)


P,L,U = linalg.lu(A)
print(P)
print(L)
print(U)


A * x = B
L (L' * x) = B
L'.conj() L y = L'.conj() L L'x

L'.conj() = y
L'.conj() = L' * x






linalg.solve(A,B) -> solves A*x=B for x
