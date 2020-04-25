# Define 3 types of DNN and count parameters

import torch
import torch.nn as nn
import torch.nn.functional as F

body_dim = 6
"""
Compare different archetectures (all input has the same dimension [?,body_dim,body_dim,body_dim,5]):

Net: Abstract
FC4: Fully Connected Layers: [input,120,84,output]
CONV2D: Conv 2D, use z dimension as channel. [input, conv2d, conv2d, 120, 84, output]
"""
class Net(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        # check input dim in x,y,z,one_hot format NHWDC
        assert list(x.size()[1:]) == [body_dim,body_dim,body_dim,5]
    
class FC4(Net):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(in_features=body_dim * body_dim * body_dim * 5, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        super().forward(x)
        x = x.view(-1, body_dim * body_dim * body_dim * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CONV2D(Net):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, padding=1) # same padding
        self.conv2 = nn.Conv2d(30, body_dim, 3, padding=1)
        self.fc1 = nn.Linear(body_dim*body_dim*body_dim, 1)

    def forward(self, x):
        super().forward(x)
        x = x.permute(0, 3,4, 1,2) # swap axes, since Torch conv2d only supports NCHW format
        x = x.view(-1, body_dim*5,body_dim,body_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2,3,1) # swap axes back from conv2d
        x = x.view(-1, body_dim*body_dim*body_dim)
        x = self.fc1(x)
        return x

class CONV3D(Net):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv3d(in_channels=5, out_channels=30, kernel_size=3, padding=1) # same padding
        self.conv2 = nn.Conv3d(30, 5, 3, padding=1) # same padding
        self.fc1 = nn.Linear(body_dim*body_dim*body_dim*5, 1)
    def forward(self, x):
        super().forward(x)
        x = x.permute(0, 4, 3, 1, 2) # swap axes, since Torch conv3d only support N,Cin,D,H,W format
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = x.permute(0, 3,4, 2,1) # swap axes back from conv2d
        x = x.view(-1, body_dim*body_dim*body_dim*5)
        x = self.fc1(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    fc = FC4()
    c2 = CONV2D()
    c3 = CONV3D()

    print("Counting parameters:")
    print("fc: ", count_parameters(fc))
    print("c2: ", count_parameters(c2))
    print("c3: ", count_parameters(c3))