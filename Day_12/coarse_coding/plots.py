from matplotlib import pyplot as plt
import numpy as np


class Plots():

    @staticmethod
    def scatterplot(y, lims=(100, 1000)):
        fig, ax = plt.subplots(figsize=(9,6))
        ax.set(xlim=(0, lims[0]), ylim=(0, lims[1]))
        x = np.array([i for i in range(1, len(y) + 1)])
        ax.plot(x,y, 'o')
        plt.savefig("scatterplot.png")
        plt.show()