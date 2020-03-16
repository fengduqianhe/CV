import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''构建神经网络'''

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channls , 5*5 square
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation y= Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  #all dimension except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

if __name__=="__main__":
    net = Net()
    print(net)
    params = list(net.parameters())
    # weight  bias
    print(len(params))
    # conv1's weight
    print(params[0].size())
    # random imput
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    net.zero_grad()
    target = torch.randn(10)
    target = target.view(1, -1)
    criterion = nn.MSELoss()
    loss = criterion(out, target)
    print(loss)
    # print(loss.grad_fn) # MSELoss
    # print(loss.grad_fn.next_functions[0][0])  # Linear
    # print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU
    # backward
    # output before and after backward  grad
    net.zero_grad()
    # print('conv1.bias.grad before backward')
    # print(net.conv1.bias.grad)
    # print(net.conv1.weight.grad)
    # print(net.fc1.weight.grad)
    # print(net.fc1.bias.grad)
    loss.backward()
    # print('conv1.bias.grad after backward')
    # print(net.conv1.bias.grad)
    # print(net.conv1.weight.grad)
    # print(net.fc1.weight.grad)
    # print(net.fc1.bias.grad)
    # weight = weight - learning_rate * gradient

    # learning_rate = 0.01
    # for f in net.parameters():
    #     # print(f.grad)
    #     print(f.data)
    #     f.data.sub_(f.grad.data * learning_rate)
    #     print(f.data)
    print(target)
    for i in  range(100):
        optimizer = optim.SGD(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print("loss", loss)
    print(output)
