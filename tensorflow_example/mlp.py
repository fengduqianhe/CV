# -*- coding: utf-8 -*-
# @Author  : qiaohezhe
# @github : https://github.com/fengduqianhe
# @Date    :  2020/1/20 12:59
# version： Python 3.7.8
# @File : mlp.py
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt

def show():
    all_x = X[:,2]
    all_y = X[:,3]
if __name__== "__main__":
    n = 0
    lr = 0.10
    X = np.array([[1, 1, 2, 3], [1, 1, 4, 5], [1, 1, 1, 1], [1, 1, 5, 3], [1, 1, 0, 1]])
    Y = np.array([1, 1, -1, 1, -1])
    W = (np.random.random(X.shape[1]) - 0.5) * 2
    for iter in range(100):
        new_output = np.sign(np.matmul(X, W.T))
        if (new_output == Y).all():
            break
        new_W = W + lr * np.matmul((Y - new_output.T), X)/int(X.shape[0])
        W = new_W
        print("第%s次迭代权重为%s" % (iter, W))

    # X = np.array([[1, 1, 2, 3, 1], [1, 1, 4, 5, 1], [-1, -1, -1, -1, -1], [1, 1, 5, 3, 1], [-1, -1, 0, -1, -1]])
    # W = np.array([0, 0, 0, 0, 0])
    # n = 0
    # while 1:
    #     flag = 0
    #     for x in X:
    #         if np.matmul(x, W.T) <= 0:
    #             W = W + x
    #             flag = 1
    #         n += 1
    #         print("第%s次迭代权重为%s" % (n, W))
    #     if flag == 0:
    #         break

# print(new_output)

