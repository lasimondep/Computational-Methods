#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'pgf.preamble': r'\usepackage[utf8x]{inputenc}' + '\n' +
        r'\usepackage[T1]{fontenc}' + '\n' +
        r'\usepackage{cmbright}'
    })


def initPlots():
    fig_NDelta = plt.figure()
    ax_NDelta = fig_NDelta.add_subplot()
    fig_rhoDelta = plt.figure()
    ax_rhoDelta = fig_rhoDelta.add_subplot()

    ax_NDelta.set_xscale('log', base=10)
    ax_NDelta.set_yscale('log', base=10)
    ax_NDelta.set_xlabel('$N$')
    ax_NDelta.set_ylabel(r'$\|\Delta\|_{l_2}$')

    ax_rhoDelta.set_xscale('log')
    ax_rhoDelta.set_yscale('log')
    ax_rhoDelta.set_xlabel('$N$')
    ax_rhoDelta.set_ylabel(r'$\rho$')

    ax_NDelta.grid()
    ax_rhoDelta.grid()

    return fig_NDelta, ax_NDelta, fig_rhoDelta, ax_rhoDelta


d = {
    'RungeKutta': 'o-',
    'CROS1': 'x-',
    'Adams': '.--',
}

for testname in ('SimpleSystem', 'LinearSystem', ):
    for schemaname in ('RungeKutta', 'CROS1', 'Adams'):
        fig_NDelta, ax_NDelta, fig_rhoDelta, ax_rhoDelta = initPlots()
        N = []
        Delta = []
        with open('./csv/NDelta_%s_%s.csv' % (testname, schemaname), 'r') as fin:
            for line in fin:
                _N, _Delta = map(float, line.strip().split(','))
                if np.isfinite(_Delta) and _Delta < 1e10:
                    N.append(_N)
                    Delta.append(_Delta)
        N = np.asarray(N)
        Delta = np.asarray(Delta)
        ax_NDelta.plot(N, Delta, d[schemaname], label=schemaname)
        ax_rhoDelta.plot(N[:-1], (-np.log(Delta[1:]) + np.log(Delta[:-1])) / np.log(2), d[schemaname], label=schemaname)
        ax_NDelta.legend()
        ax_rhoDelta.legend()
        fig_NDelta.savefig('./img/NDelta_%s_%s.png' % (testname, schemaname))
        fig_rhoDelta.savefig('./img/rhoDelta_%s_%s.png' % (testname, schemaname))
            
