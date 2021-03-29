import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': r'\usepackage[utf8x]{inputenc}' + '\n' +
        r'\usepackage[T1]{fontenc}' + '\n' +
        r'\usepackage{cmbright}'
    })


class Plot2D:
    def __init__(self, label_axes=('', ''), log_scale=(False, False)):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot()

        self.ax.set_xlabel(label_axes[0])
        self.ax.set_ylabel(label_axes[1])
        if log_scale[0]:
            self.ax.set_xscale('log', base=10)
        if log_scale[1]:
            self.ax.set_yscale('log', base=10)
        self.ax.grid()

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)
        self.ax.legend()

    def save(self, filename):
        self.figure.savefig(filename)

    def show(self):
        plt.tight_layout()
        plt.show()

class Plot3D:
    def __init__(self, label_axes=('', '', '')):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(projection='3d')

        self.ax.set_xlabel(label_axes[0])
        self.ax.set_ylabel(label_axes[1])
        self.ax.set_zlabel(label_axes[2])

    def plot(self, *args, **kwargs):
        self.ax.plot_surface(*args, cmap=cm.coolwarm, **kwargs)

    def show(self):
        plt.tight_layout()
        plt.show()
