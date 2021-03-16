import numpy as np
import matplotlib.pyplot as plt


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
        plt.show()
