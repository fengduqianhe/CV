'''利用SVD对图片进行重构，对比重构后的图片'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
'''加载图片'''
img_eg = mpimg.imread("./image/image1.jpg")
print(img_eg.shape)
img_temp = img_eg.reshape(640, 640*3)
print(img_temp.shape)
U, Sigma, VT = np.linalg.svd(img_temp)
# print(Sigma)
'''取前60个奇异值'''
sval_nums = 10
img_restruct1 = U[:, 0:sval_nums]@np.diag(Sigma[0:sval_nums])@VT[0:sval_nums, :]
print(img_restruct1.shape)

'''取前60个奇异值'''
sval_nums = 120
img_restruct2 = U[:, 0:sval_nums]@np.diag(Sigma[0:sval_nums])@VT[0:sval_nums, :]
print(img_restruct2.shape)


'''重构图片'''
img_restruct1 = img_restruct1.reshape(640, 640, 3)
img_restruct2 = img_restruct2.reshape(640, 640, 3)


fig, ax = plt.subplots(1, 3, figsize=(50, 32))
ax[0].imshow(img_eg)
ax[0].set(title="src")
ax[1].imshow(img_restruct1.astype(np.uint8))
ax[1].set(title="nums of sigma = 10")
ax[2].imshow(img_restruct2.astype(np.uint8))
ax[2].set(title="nums of sigma = 120")

fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(Sigma)
axes[1].plot(Sigma.cumsum())

plt.show()


