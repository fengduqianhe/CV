import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''构建图像分类器'''
class Net(nn.Module):
    #build structer
    def __init__(self):
        super(Net, self).__init__()
