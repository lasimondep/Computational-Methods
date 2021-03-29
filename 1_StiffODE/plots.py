#!/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d



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

    #ax_NDelta.set_xscale('log', base=10)
    ax_NDelta.set_yscale('log', base=10)
    ax_NDelta.set_xlabel('$N$')
    ax_NDelta.set_ylabel(r'$\|\Delta\|_{l_2}$')

    ax_rhoDelta.set_xscale('log')
    #ax_rhoDelta.set_yscale('log')
    ax_rhoDelta.set_xlabel('$N$')
    ax_rhoDelta.set_ylabel(r'$\rho$')

    ax_NDelta.grid()
    ax_rhoDelta.grid()

    return fig_NDelta, ax_NDelta, fig_rhoDelta, ax_rhoDelta


d = {
    'RungeKutta': 'o',
    'CROS1': 'o',
    'Adams': 'o',
}

for testname in ('LinearSystem', 'AutonomousSystem', 'ConstEigenSystem'):
    fig_NDelta, ax_NDelta, fig_rhoDelta, ax_rhoDelta = initPlots()
    for schemaname in ('RungeKutta', 'CROS1', 'Adams'):
        #fig_NDelta, ax_NDelta, fig_rhoDelta, ax_rhoDelta = initPlots()
        N = []
        Delta = []
        t = []
        u = []
        with open('./csv/NDelta_%s_%s.csv' % (testname, schemaname), 'r') as fin:
            for line in fin:
                _N, _Delta = map(float, line.strip().split(','))
                if np.isfinite(_Delta) and _Delta < 1e10:
                    N.append(_N)
                    Delta.append(_Delta)
        N = np.asarray(N)
        Delta = np.asarray(Delta)

        #_k = 0
        #with open('./csv/tu_%s_%s.csv' % (testname, schemaname), 'r') as fin:
        #    for line in fin:
        #        input_line = list(map(float, line.strip()[:-1].split(',')))
        #        if _k % 10 == 0:
        #            t.append(input_line[0])
        #            u.append(input_line[1:])
        #        _k += 1


        #t = np.asarray(t)
        #u = np.asarray(u)
        #_t, x = np.meshgrid(t, np.arange(len(u)))
        #fig_tu = plt.figure()
        #ax_tu = fig_tu.add_subplot(projection='3d')
        #ax_tu.plot(x, _t, u)
        #plt.show()

        ax_NDelta.plot(N, Delta, d[schemaname], label=schemaname)
        ax_rhoDelta.plot(N[:-1], (np.log(Delta[1:]) - np.log(Delta[:-1])) / np.log(2), d[schemaname], label=schemaname)
        ax_NDelta.legend()
        ax_rhoDelta.legend()
        #fig_NDelta.savefig('./img/NDelta_%s_%s.png' % (testname, schemaname))
        fig_NDelta.savefig('./img/NDelta_%s.png' % testname)
        #fig_rhoDelta.savefig('./img/rhoDelta_%s_%s.png' % (testname, schemaname))
        fig_rhoDelta.savefig('./img/rhoDelta_%s.png' % testname)
            
