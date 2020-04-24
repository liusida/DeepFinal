# Step 6. see the relationship in rank between truth and prediction
# Require: model file in models/ folder.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time, sys, math

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

names = ["FC4" , "CONV2D", "CONV3D"]
train_colors = {"FC4":"#001f2c" , "CONV2D":"#8c2060", "CONV3D":"#ef7600"}
test_colors = {"FC4":"#005f7c" , "CONV2D":"#bc5090", "CONV3D":"#ffa600"}
fig,ax = plt.subplots(1,3,sharex=True, sharey=True, figsize=[10,3])

for idx, name in enumerate(names):
    model = torch.load(f"models/{name}_0.model")
    model.eval()

    arg = torch.argsort(test_Y, 0).view(-1)
    test_X = test_X[arg]
    test_Y = test_Y[arg]
    Y_hat = model(test_X)
    arg = torch.argsort(Y_hat, 0).view(-1).cpu().numpy()
    print(arg)

    ax[idx].scatter( list(range(len(arg))), arg, c=train_colors[name], s=0.5 )
    ax[idx].set_title(name)

    if idx==1:
        ax[idx].set_xlabel("Rank in Truth")
    if idx==0:
        ax[idx].set_ylabel("Rank in Prediction")

plt.tight_layout()
plt.savefig("plots/compare_ranks.svg")