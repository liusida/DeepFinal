# Step 2. Compare testing data to all the training data, delete the testing data that is too similar to one of the training data for more rigorous testing.
# this will produce a search.npy in cache to save comparing time
# this will produce two npy files in data/cleaned_test_data/ for further use.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time, sys
import numpy as np

import preprocess
import datasets

torch.manual_seed(1)

GPU = True
if GPU:
    dtype = torch.cuda.FloatTensor
    device = 'GPU'
else:
    dtype = torch.FloatTensor
    device = 'RAM'
if False:
    print(f"Loading training data to {device}...")
    train_X, train_Y = preprocess.load_data_torch(datasets.training_data, dtype)
    print(f"Loading testing data to {device}...")
    test_X, test_Y = preprocess.load_data_torch(datasets.testing_data, dtype)

    a = train_X.view(-1,6*6*6*5)
    b = test_X.view(-1,6*6*6*5)

    search = []
    for i in range(b.size()[0]):
        c = (a == b[i])
        c = torch.sum(c, 1)
        c = torch.max(c)
        if c==1080:
            print(c.item(), end=" ", flush=True)
        search.append(c.item())

    search =np.array(search)
    np.save(file="cache/search", arr=search)
else:
    search = np.load("cache/search.npy")

test_X, test_Y = preprocess.load_data_torch(datasets.testing_data, dtype)

# include criteria: at least 80 features different from any existing one. (total 1080 features). 1000=1080-80
test_X = test_X[search<1000]
test_Y = test_Y[search<1000]
print(test_X.size())
print(test_Y.size())

test_X = test_X.cpu().numpy()
np.save(file="data/cleaned_test_data/test_X", arr=test_X)
test_Y = test_Y.cpu().numpy()
np.save(file="data/cleaned_test_data/test_Y", arr=test_Y)
