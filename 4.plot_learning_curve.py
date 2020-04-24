# Step 4. plot learning curve and compare test loss across different settings.
# Require: results saved in results/ folder.

import numpy as np
import matplotlib.pyplot as plt

names = ["FC4" , "CONV2D", "CONV3D"]
train_colors = {"FC4":"#001f2c" , "CONV2D":"#8c2060", "CONV3D":"#ef7600"}
test_colors = {"FC4":"#005f7c" , "CONV2D":"#bc5090", "CONV3D":"#ffa600"}
fig,ax = plt.subplots(1,3,sharex=True, sharey=True, figsize=[10,3])

for idx,name in enumerate(names):
    train_y = []
    test_y = []
    for i in range(5):
        train_y.append(np.load(f"results/train_loss_{name}_{i}.npy"))
        test_y.append(np.load(f"results/test_loss_{name}_{i}.npy"))
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    train_y_mean = np.mean(train_y,axis=0)
    train_y_std = np.std(train_y,axis=0)
    train_y_min = train_y_mean - train_y_std
    train_y_max = train_y_mean + train_y_std
    test_y_mean = np.mean(test_y,axis=0)
    test_y_std = np.std(test_y,axis=0)
    test_y_min = test_y_mean - test_y_std
    test_y_max = test_y_mean + test_y_std

    x = list(range(len(train_y_mean)))
    ax[idx].plot(x, test_y_mean, label=f"test", c=test_colors[name], alpha=1)
    ax[idx].plot(x, train_y_mean, label=f"train", c=train_colors[name], alpha=1)
    ax[idx].fill_between(x, test_y_min, test_y_max, color=test_colors[name], alpha=0.2)
    ax[idx].fill_between(x, train_y_min, train_y_max, color=train_colors[name], alpha=0.2)

    ax[idx].set_ylim((0.000,0.001))
    ax[idx].set_title(name)
    if idx==0:
        ax[idx].set_ylabel("MSE")
    if idx==1:
        ax[idx].set_xlabel("# of Epoch")
    
    ax[idx].legend()

plt.tight_layout()
plt.legend()
plt.savefig("plots/learning_curves.svg")