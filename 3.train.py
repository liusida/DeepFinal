# Step 3. Train the networks and test
# Require: cleaned test data stored in data/cleaned_test_data/ folder
# this file should be run multiple times
# producing results of combinations of network types and random seeds in results/ folder
# producing models in models/ folder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time, sys, math
import numpy as np

# my packages
import preprocess
import datasets
import visualize
import networks

if len(sys.argv)<3:
    print(f"Usage:\n python main.py <[0-2]> <seed>\n\n0: Fully Connected\n1: Convolutional 2D\n2: Convolutional 3D\n")
    net_id = 0
    random_seed = 0
else:
    net_id = int(sys.argv[1])
    random_seed = int(sys.argv[2])

torch.manual_seed(random_seed)

GPU = True
if GPU:
    dtype = torch.cuda.FloatTensor
    device = 'GPU'
else:
    dtype = torch.FloatTensor
    device = 'RAM'

print(f"Loading training data to {device}...")
train_X, train_Y = preprocess.load_data_torch(datasets.training_data, dtype)
print(f"Loading testing data to {device}...")
# test_X, test_Y = preprocess.load_data_torch(datasets.testing_data, dtype)
test_X, test_Y = preprocess.load_cleaned_test_data_torch(dtype)
champion = torch.argmax(test_Y)
losser = torch.argmin(test_Y)
visualize.visualize_robot(test_X[champion].cpu().numpy(), "test_champion.png")
visualize.visualize_robot(test_X[losser].cpu().numpy(), "test_losser.png")

print(f"Training X Shape: {train_X.shape}")
print(f"Testing X Shape: {test_X.shape}")
print("",flush=True)

nets = [
    networks.FC4(),
    networks.CONV2D(),
    networks.CONV3D(),
]
base_num = 400
epochs = {
    nets[0]:base_num,
    nets[1]:base_num,
    nets[2]:base_num,
}
nets_name = {
    nets[0]:"FC4", 
    nets[1]:"CONV2D",
    nets[2]:"CONV3D",
}

# for net in nets:
if True:
    net = nets[net_id]
    start_time = time.time()
    print(f"Training {nets_name[net]}...")
    if GPU:
        net.cuda()
    criterion = nn.MSELoss()
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    batch_size = 512
    total_number = train_X.size()[0]
    
    train_loss = []
    test_loss = []
    # TRAIN
    for epoch in range(epochs[net]):
        for i in range(math.ceil(total_number/batch_size)):
            batch_start = i*batch_size
            batch_end = (i+1)*batch_size if (i+1)*batch_size<total_number-1 else total_number-1
            # print(f"{batch_start} - {batch_end}")
            train_X_batch = train_X[batch_start:batch_end]
            train_Y_batch = train_Y[batch_start:batch_end]

            optimizer.zero_grad()   # zero the gradient buffers
            Y_hat = net(train_X_batch)
            loss = criterion(Y_hat, train_Y_batch)
            loss.backward()
            optimizer.step()    # Does the update
            msg = ''

        if epoch%1==0:
            msg += f"epoch: {epoch:05}; train_loss: {loss:.5f}; "
            train_loss.append(float(loss))
            # TEST
            Y_hat = net(test_X)
            loss = criterion(Y_hat, test_Y)
            
            test_loss.append(float(loss))
            msg += f"test_loss: {loss:.5f}; champion: P {Y_hat[champion].data[0]:.3f}, T {test_Y[champion].data[0]:.3f}; losser: P {Y_hat[losser].data[0]:.3f}, T {test_Y[losser].data[0]:.3f} "
            print(msg, end='\r', flush=True)

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)
    np.save(f"results/train_loss_{nets_name[net]}_{random_seed}", train_loss)
    np.save(f"results/test_loss_{nets_name[net]}_{random_seed}", test_loss)

    x = torch.arange(len(train_loss))
    plt.figure(figsize=[9,6])
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, test_loss, label="test_loss")
    plt.ylim((0,0.001))
    plt.legend()
    plt.savefig(f"{nets_name[net]}_{random_seed}.png")
    print(f"\n{nets_name[net]} finished. {(time.time() - start_time):.2f} seconds.\n")

    torch.save(net, f"models/{nets_name[net]}_{random_seed}.model")

print("\n")