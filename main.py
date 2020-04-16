import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time, sys

# my packages
import preprocess
import datasets
import visualize
import networks

if len(sys.argv)<2:
    print(f"Usage:\n python main.py [0-2]\n\n0: Fully Connected\n1: Convolutional 2D\n2: Convolutional 3D\n")
    exit(0)
net_id = int(sys.argv[1])

torch.manual_seed(1)
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
test_X, test_Y = preprocess.load_data_torch(datasets.testing_data, dtype)
champion = torch.argmax(test_Y)
print(f"champion: {champion}")
print(test_X[champion].shape)
print(test_Y[champion])

visualize.visualize_robot(test_X[champion].cpu().numpy(), "test_champion.png")

print(f"Training X Shape: {train_X.shape}")
print(f"Testing X Shape: {test_X.shape}")
print("",flush=True)

nets = [
    networks.FC4(),
    networks.CONV2D(),
    networks.CONV3D(),
]
base_num = 10000
epochs = {
    nets[0]:base_num * 30,
    nets[1]:base_num * 3,
    nets[2]:base_num,
}

# for net in nets:
if True:
    net = nets[net_id]
    start_time = time.time()
    print(f"Training {net.__class__}...")
    if GPU:
        net.cuda()
    criterion = nn.MSELoss()
    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    train_loss = []
    test_loss = []
    # TRAIN
    for epoch in range(epochs[net]):
        optimizer.zero_grad()   # zero the gradient buffers
        Y_hat = net(train_X)
        loss = criterion(Y_hat, train_Y)
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
            msg += f"test_loss: {loss:.5f}; champion: pred {Y_hat[champion].data[0]:.4f}, truth {test_Y[champion].data[0]:.4f}."
            print(msg, end='\r', flush=True)


    x = torch.arange(len(train_loss)) * 10
    plt.figure(figsize=[9,6])
    plt.plot(x, train_loss, label="train_loss")
    plt.plot(x, test_loss, label="test_loss")
    plt.legend()
    plt.savefig(f"{net.__class__}.png")
    print(f"\n{net.__class__} finished. {(time.time() - start_time):.2f} seconds.\n")

print("\n")