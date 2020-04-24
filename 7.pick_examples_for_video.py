# Step 6. see the relationship in rank between truth and prediction
# Require: model file in models/ folder.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time, sys, math, os
import numpy as np
from lxml import etree

# my packages
import preprocess
import datasets
import visualize
import networks

torch.manual_seed(1)
GPU = True
if GPU:
    dtype = torch.cuda.FloatTensor
    device = 'GPU'
else:
    dtype = torch.FloatTensor
    device = 'RAM'
test_X, test_Y = preprocess.load_cleaned_test_data_torch(dtype)

names = ["FC4"]
train_colors = {"FC4":"#001f2c" , "CONV2D":"#8c2060", "CONV3D":"#ef7600"}
test_colors = {"FC4":"#005f7c" , "CONV2D":"#bc5090", "CONV3D":"#ffa600"}
plt.figure(figsize=[4,4])

for idx, name in enumerate(names):
    model = torch.load(f"models/{name}_0.model")
    model.eval()

    arg = torch.argsort(test_Y, 0).view(-1)
    test_X = test_X[arg]
    test_Y = test_Y[arg]
    Y_hat = model(test_X)
    arg = torch.argsort(Y_hat, 0).view(-1).cpu().numpy()
    print(arg)

    plt.scatter( list(range(len(arg))), arg, c=test_colors[name], s=0.5, alpha=0.3)

    example = [10, 20, 100, 800, 1000, 1200, 1400, 1820, 1830]
    example = range(1,1841,100)
    for i in example:
        # print(test_X[i].size(), test_Y[i], Y_hat[i], arg[i])
        # robot = torch.argmax(test_X[i], 3)
        print(f"Recording robot_{i}")
        robot = test_X[i].cpu().numpy()
        print(test_Y[i])
        visualize.visualize_robot(robot, f"video/robot_{i}_{arg[i]}.svg")
        plt.scatter( i, arg[i], c=train_colors[name], s=10.0, alpha=1)
        robot = np.argmax(robot, 3)
        robot = np.swapaxes(robot, 0, 1)
        robot = robot.reshape(36,6)
        vxd = etree.Element("VXD")
        Lattice_Dim = etree.SubElement(vxd, "Lattice_Dim")
        Lattice_Dim.set("replace", "VXA.VXC.Lattice.Lattice_Dim")
        Lattice_Dim.text = "0.016666666666666666"
        Structure = etree.SubElement(vxd, "Structure")
        Structure.set("replace", "VXA.VXC.Structure")
        Structure.set("Compression", "ASCII_READABLE")
        etree.SubElement(Structure, "X_Voxels").text="6"
        etree.SubElement(Structure, "Y_Voxels").text="6"
        etree.SubElement(Structure, "Z_Voxels").text="6"
        Data = etree.SubElement(Structure, "Data")
        for layer in range(6):
            etree.SubElement(Data, "Layer").text = "".join([str(c) for c in robot[:,layer]])
        try:
            os.system(f"rm video/data/*.vxd")
        except:
            pass
        with open(f'video/data/robot_{i}.vxd', 'wb') as f:
            f.write(etree.tostring(vxd))
        # os.system(f"./bin/Voxelyze3 -w ./bin/vx3_node_worker -i video/data > video/robot_{i}.history")
        # os.system(f"mv video/data/robot_{i}.vxd video/")



plt.xlabel("Rank in Truth")
plt.ylabel("Rank in Prediction")
plt.xticks([0,500,1000,1500])
plt.yticks([0,500,1000,1500])
plt.show()