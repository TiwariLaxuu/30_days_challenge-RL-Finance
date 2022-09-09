import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math

class Animation():
    
    def __init__(self, starting_pos, positions):
        self.x = np.linspace(-1.2, 0.6, 100)
        self.positions = positions
        self.fig, ax = plt.subplots(figsize=(9,6))
        ax.set(xlim=(-1.3, 0.7), ylim=(-1.2, 1.2))
        ax.plot(self.x, self.y_func(self.x))
        self.point, = ax.plot(starting_pos, self.y_func(starting_pos), 'o')

    def plot_curve(self):
        ani = FuncAnimation(self.fig, self.update, interval=10, blit=True, repeat=True, frames=self.positions)
        ani.save("animation.gif", writer="imagemagick", fps=60)
        plt.show()

    def update(self, x):
        y = self.y_func(x)
        self.point.set_data([x], [y])
        return self.point, 

    def y_func(self, x):
        return np.cos(3*(x + math.pi/2))
