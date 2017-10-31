import numpy as np
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
from math import ceil, floor, log10

cal_type = 'Calibrated'
sig_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_' + cal_type + '_Hists/'
obs_pathlist = glob.glob(sig_path + '*sigma*')
T_cont_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_temp_avg_cont.npy'
T_rxr_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_temp_avg_rxr.npy'
pols = ['xx', 'yy', 'xy', 'yx']
N_freqs = 896
f_chan = 458
outpath = '/Users/mike_e_dubs/HERA/Temperatures/GS_' + cal_type + '_Plots/'
rb = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
c = []
for m in range(6):
    c += 12 * [rb[m]]


temp_avg_cont = np.load(T_cont_path)[:-1]
temp_avg_rxr = np.load(T_rxr_path)[:-1]
x = np.linspace(min(temp_avg_cont) - 1, max(temp_avg_cont) + 1, num=300)
y = np.linspace(min(temp_avg_rxr) - 1, max(temp_avg_rxr) + 1, num=300)
X, Y = np.meshgrid(x, y)

sigma = {}
sigma_corr_rxr = {}  # Will be plotted against temp_avg_rxr
sigma_corr_cont = {}  # Will be plotted against temp_avg_cont
for pol in pols:
    sigma[pol] = np.zeros([len(obs_pathlist) / 4, N_freqs])
    sigma_corr_rxr[pol] = np.zeros(len(obs_pathlist) / 4 - 1)
    sigma_corr_cont[pol] = np.zeros(len(obs_pathlist) / 4 - 1)

k = 0
for path in obs_pathlist:
    obs = path[path.find('zen'):path.find('.HH') + 3]
    pol = obs[-5:-3]
    sigma[pol][k / 4, :] += np.load(sig_path + obs + '_sigma_' + pol.upper() + '.npy')
    k += 1

for pol in pols:

    A_cont = np.c_[temp_avg_cont, np.ones(len(temp_avg_cont))]
    C_cont, res_cont, _, _ = scipy.linalg.lstsq(A_cont, sigma[pol][:-1, f_chan])
    Z_cont = C_cont[0] * x + C_cont[1]

    A_rxr = np.c_[temp_avg_rxr, np.ones(len(temp_avg_rxr))]
    C_rxr, res_rxr, _, _ = scipy.linalg.lstsq(A_rxr, sigma[pol][:-1, f_chan])
    Z_rxr = C_rxr[0] * y + C_rxr[1]

    A = np.c_[temp_avg_cont, temp_avg_rxr, np.ones(len(temp_avg_cont))]
    C, res, _, _ = scipy.linalg.lstsq(A, sigma[pol][:-1, f_chan])  # :-1 since last temp writeout not averaged properly
    Z = C[0] * X + C[1] * Y + C[2]

    x0 = np.mean(temp_avg_cont)
    y0 = np.mean(temp_avg_rxr)
    dzx = np.array([C[0] * (x0 - T) for T in temp_avg_cont])
    dzy = np.array([C[1] * (y0 - T) for T in temp_avg_rxr])

    sigma_corr_rxr[pol] = sigma[pol][:-1, f_chan] + dzx
    sigma_corr_cont[pol] = sigma[pol][:-1, f_chan] + dzy

    C_cont_corr, res_cont_corr, _, _ = scipy.linalg.lstsq(A_cont, sigma_corr_cont[pol])
    Z_cont_corr = C_cont_corr[0] * x + C_cont_corr[1]

    C_rxr_corr, res_rxr_corr, _, _ = scipy.linalg.lstsq(A_rxr, sigma_corr_rxr[pol])
    Z_rxr_corr = C_rxr_corr[0] * y + C_rxr_corr[1]

    fig_title = pol.upper() + ' Sigma (' + cal_type + ') vs. Ambient Temperature, f = 151 Mhz'
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
    fig.suptitle(fig_title)
    ax[0].scatter(temp_avg_cont, sigma[pol][:-1, f_chan], c=c)
    ax[0].plot(x, Z_cont, label='C = ' + str(['%.3E' % (value) for value in C_cont]) +
               ', res = ' + '%.3E' % (res_cont))
    ax[0].set_ylabel('Sigma')
    ax[0].set_xlabel('Container Temperature (K)')
    ax[0].legend()
    ax[1].scatter(temp_avg_rxr, sigma[pol][:-1, f_chan], c=c)
    ax[1].plot(y, Z_rxr, label='C = ' + str(['%.3E' % (value) for value in C_rxr]) +
               ', res = ' + '%.3E' % (res_rxr))
    ax[1].set_ylabel('Sigma')
    ax[1].set_xlabel('Receiver 5 Temperature (K)')
    ax[1].legend()

    fig3d = plt.figure(figsize=(14, 8))
    ax3d = fig3d.add_subplot(111, projection='3d')
    fig3d.suptitle(fig_title + ', C = ' + str(['%.3E' % (value) for value in C]) +
                   ', res = ' + '%.3E' % (res))
    ax3d.scatter(temp_avg_cont, temp_avg_rxr, sigma[pol][:-1, f_chan], c=c)
    ax3d.plot_surface(X, Y, Z, alpha=0.2)
    ax3d.set_xlabel('Container Temperature (K)')
    ax3d.set_ylabel('Receiver 5 Temperature (K)')
    ax3d.set_zlabel('Sigma')

    fig_corr, ax_corr = plt.subplots(figsize=(14, 8), nrows=2)
    fig_corr.suptitle(pol.upper() + ' Corrected Sigma (' + cal_type + ') vs. Ambient Temperature, f = 151 Mhz')
    ax_corr[0].scatter(temp_avg_cont, sigma_corr_cont[pol], c=c)
    ax_corr[0].plot(x, Z_cont_corr, label='C = ' + str(['%.3E' % (value) for value in C_cont_corr]) +
                    ', res = ' + '%.3E' % (res_cont_corr))
    ax_corr[0].set_xlabel('Container Temperature (K)')
    ax_corr[0].set_ylabel('Sigma (Corrected)')
    ax_corr[0].legend()
    ax_corr[1].scatter(temp_avg_rxr, sigma_corr_rxr[pol], c=c)
    ax_corr[1].plot(y, Z_rxr_corr, label='C = ' + str(['%.3E' % (value) for value in C_rxr_corr]) +
                    ', res = ' + '%.3E' % (res_rxr_corr))
    ax_corr[1].set_xlabel('Receiver 5 Temperature (K)')
    ax_corr[1].set_ylabel('Sigma (Corrected)')
    ax_corr[1].legend()

    fig.savefig(outpath + 'Temp_Corr_' + pol + '.png')
    fig3d.savefig(outpath + 'Temp_Corr_3d_' + pol + '.png')
    fig_corr.savefig(outpath + 'Temp_Corr_Correction_' + pol + '.png')
