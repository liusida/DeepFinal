# Step 5. see the relationship in rank between truth and prediction
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

model = torch.load("models/fc.model")
model.eval()

arg = torch.argsort(test_Y, 0).view(-1)
test_X = test_X[arg]
test_Y = test_Y[arg]
Y_hat = model(test_X)
arg = torch.argsort(Y_hat, 0).view(-1).cpu().numpy()
print(arg)

plt.figure(figsize=[6,6])
plt.scatter( list(range(len(arg))), arg, s=0.5 )
plt.xlabel("Rank in Truth")
plt.ylabel("Rank in Prediction")
plt.show()