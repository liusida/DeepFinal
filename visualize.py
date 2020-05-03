
import preprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

def visualize_robot(body, png_filename='', swapaxes=False):
    """
    params:
        body: a [6,6,6,5] numpy array, defining a 6x6x6 body with one-hot encoding.
    see also: function  body_one_hot() in 8.evolution.py to convert [?,6,6,6] to [?,6,6,6,5]
    """
    import numpy as np
    x, y, z = np.indices((6, 6, 6))
    voxels = np.zeros_like(x, dtype=bool)
    colors = np.empty(voxels.shape, dtype=object)
    body = np.argmax(body, axis=len(body.shape)-1)
    if swapaxes:
        body = np.swapaxes(body,0,2)
    voxels = body > 0
    colors[body==0] = '#000000'
    colors[body==1] = '#00ffff'
    colors[body==2] = '#0000ff'
    colors[body==3] = '#ff0000'
    colors[body==4] = '#00ff00'
    # and plot everything
    fig = plt.figure(figsize=[6.6,6])
    ax = fig.gca(projection='3d')
    ax.set_proj_type('persp')
    ax.voxels(voxels, facecolors=colors, edgecolor='k', linewidth=0.1, alpha=1)
    plt.axis('off')
    if png_filename=='':
        plt.show()
    else:
        plt.savefig(png_filename)
    plt.close()

if __name__ == "__main__":
    dataset = [
        ["data/BBTwo1000/generation_0000/report/output.xml",
        "data/BBTwo1000/generation_0000/start_population"],
    ]
    test_X, test_Y = preprocess.load_data_torch(dataset)

    visualize_robot(test_X[0].numpy(), png_filename='tmp.png')