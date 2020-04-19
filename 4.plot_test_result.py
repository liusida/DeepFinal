# Step 4. plot learning curve and compare test loss across different settings.
# Require: results saved in results/ folder.

import numpy as np
import matplotlib.pyplot as plt

test_loss_FC4_0 = np.load("results/test_loss_FC4_0.npy")
test_loss_CONV2D_0 = np.load("results/test_loss_CONV2D_0.npy")
test_loss_CONV3D_0 = np.load("results/test_loss_CONV3D_0.npy")

x = list(range(len(test_loss_FC4_0)))
plt.plot(x, test_loss_FC4_0, label="FC4")

x = list(range(len(test_loss_CONV2D_0)))
plt.plot(x, test_loss_CONV2D_0, label="CONV2D")

x = list(range(len(test_loss_CONV3D_0)))
plt.plot(x, test_loss_CONV3D_0, label="CONV3D")

plt.xlabel("# of Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()