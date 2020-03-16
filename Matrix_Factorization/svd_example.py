import numpy as np
from numpy.linalg import svd
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

A = np.array([[1, 5, 7, 6, 1], [2, 1, 10, 4, 4], [3, 6, 7, 5, 2]])
AA_T = np.matmul(A, A.T)
A_TA = np.matmul(A.T, A)

# print("A=", A)
# print("AA_T=", AA_T)
# print("A_TA=", A_TA)

'''特征值分解'''
e_vals_1, e_vecs_1 = np.linalg.eig(AA_T)
# print("e_vals of AA_T", e_vals_1)
# print("e_vecs of AA_T", e_vecs_1)

e_vals_2, e_vecs_2 = np.linalg.eig(A_TA)
# print("e_vals of A_TA", e_vals_2)
# print("e_vecs of A_TA", e_vecs_2)

'''验证特征值分解'''
lamda = np.diag(e_vals_1)
# print(lamda)
# print(e_vecs_1@lamda@e_vecs_1.T)
# print(np.sqrt(lamda))

'''奇异值分解'''
u, s, v_h = np.linalg.svd(A)
print(u)
print(s)
print(v_h)
'''验证'''
sigma = np.diag(s, k=-2)
sigma = sigma[2:, :]
print(sigma)
print(u@sigma@v_h)


