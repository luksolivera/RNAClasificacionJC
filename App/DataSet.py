import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

class DataSet:
    def getDataset(shape, factor,des ):
        X, Y = make_circles(n_samples=shape, factor=factor, noise=des)
        return (X,Y,2)

    def show(X,Y):
        plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
        plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
        plt.axis("equal")
        plt.show()