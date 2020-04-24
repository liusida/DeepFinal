# Step 5. plot learning curve and compare test loss across different settings.
# Require: results saved in results/ folder.

import numpy as np
import matplotlib.pyplot as plt

names = ["FC4" , "CONV2D", "CONV3D"]
colors = {"FC4":"#003f5c" , "CONV2D":"#bc5090", "CONV3D":"#ffa600"}
for name in names:
    y = []
    for i in range(5):
        y.append(np.load(f"results/test_loss_{name}_{i}.npy"))
    y = np.array(y)
    print(y.shape)
    y_mean = np.mean(y,axis=0)
    y_std = np.std(y,axis=0)
    y_min = y_mean - y_std
    y_max = y_mean + y_std
    print(y_mean.shape)
    x = list(range(y_mean.shape[0]))
    plt.plot(x, y_mean, label=f"{name}", c=colors[name], alpha=1)
    plt.fill_between(x, y_min, y_max, color=colors[name], alpha=0.2)
plt.ylim((0.0004,0.001))
plt.xlabel("# of Epoch")
plt.ylabel("MSE")
plt.legend()
plt.savefig("plots/compare_across_architecture.svg")