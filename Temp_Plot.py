import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata
from math import floor, ceil, log10
from matplotlib.ticker import AutoMinorLocator
import glob

cal_title = 'Uncalibrated'
inpath = '/Users/mike_e_dubs/HERA/Temperatures/GS_' + cal_title + '_Hists/'
obs_pathlist = glob.glob(inpath + '*bins.npy')  # Might want to just make the obs_list with shell script in a txt file
outpath = '/Users/mike_e_dubs/HERA/Temperatures/GS_' + cal_title + '_Plots/'
N_freqs_removed = 128
UV = pyuvdata.UVData()
freq_array_obs_path = '/Users/mike_e_dubs/HERA/Data/miriad/temp_HERA_data/zen.2457555.40356.xx.HH.uvc'
UV.read_miriad(freq_array_obs_path)
N_freqs = UV.Nfreqs - N_freqs_removed
freq_chan_interval = [64, 960]
freq_array = UV.freq_array[0, min(freq_chan_interval):max(freq_chan_interval)]
curve_calc = True
ratio = True
ratio_outpath = '/Users/mike_e_dubs/HERA/Temperatures/GS_Ratio_Data/'
pols = ['XX', 'YY', 'XY', 'YX']
hist_label = 'Histogram ("Unflagged" Data Only)'
hist_title = 'Visibility Difference Histogram, HERA Golden Set, Band Edges/Autos Removed, ' + cal_title
waterfall_title = 'HERA Golden Set Temperatures, Band Edges/Autos Removed, ' + cal_title
hist_fig_name = 'HERA_GS_' + cal_title + '_Autos_Edges_Hist_Fit.png'
temp_fig_name = 'HERA_GS_' + cal_title + '_Autos_Edges_Temperatures.png'
cu_fig_name = 'HERA_GS_' + cal_title + '_Autos_Edges_Curves.png'
curve_freqs = [160, 346, 458, 690, 881]
LST = True
if LST:
    cu_fig_name = 'HERA_GS_' + cal_title + '_LST_Autos_Edges_Curves.png'


def sigfig(x, s=4):  # s is number of sig-figs
    if x == 0:
        return(0)
    else:
        n = int(floor(log10(np.absolute(x))))
        y = 10**n * round(10**(-n) * x, s - 1)
        return(y)


sigma = {}
for pol in pols:
    sigma[pol] = np.zeros([len(obs_pathlist) / 4, N_freqs])

k = 0
for path in obs_pathlist:
    obs = path[path.find('zen'):path.find('.HH') + 3]
    pol = obs[-5:-3].upper()
    if k == 0:
        n = np.load(inpath + obs + '.u_hist.npy')
        bins = np.load(inpath + obs + '.u_bins.npy')
        fit = np.load(inpath + obs + '.u_fit.npy')
        sigma[pol][k / 4, :] += np.load(inpath + obs + '.u_sigma_' + pol + '.npy')
    else:
        n += np.load(inpath + obs + '.u_hist.npy')
        fit += np.load(inpath + obs + '.u_fit.npy')
        sigma[pol][k / 4, :] += np.load(inpath + obs + '.u_sigma_' + pol + '.npy')
    k += 1

if ratio:
    for pol in pols:
        np.save(ratio_outpath + cal_title + '_Sigma_' + pol + '.npy', sigma[pol])

# count = {}

# for pol in pols:
#    count[pol] = []
# for pol in pols:
#    for f in range(len(freq_array)):
#        if float(np.count_nonzero(sigma[pol][:, f])) / len(sigma[pol][:, f]) > 0.95:
#            count[pol].append(f)

residual = n - fit

widths = np.diff(bins)
centers = bins[:-1] + 0.5 * widths
N = len(n)

hist_fig, hist_ax = plt.subplots(figsize=(14, 8), nrows=2)
hist_ax[0].step(bins[:-1], n, where='pre', label=hist_label)
hist_ax[0].plot(centers, fit, label='Fit')
hist_ax[0].set_title(hist_title)
hist_ax[0].set_xlabel('Amplitude')
hist_ax[0].set_ylabel('Counts')
hist_ax[0].set_xscale('log', nonposy='clip')
hist_ax[0].set_yscale('log', nonposy='clip')
hist_ax[0].set_ylim([10 ** (-1), 10 * max(n)])
hist_ax[0].legend()
hist_ax[1].plot(centers, residual, label='Residual')
hist_ax[1].set_xscale('log', nonposy='clip')
hist_ax[1].set_yscale('linear')
hist_ax[1].set_xlabel('Amplitude')
hist_ax[1].set_ylabel('Counts')
hist_ax[1].legend()

temp_fig, temp_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
temp_fig.suptitle(waterfall_title)
auto_pol_max = max([np.amax(sigma['XX']), np.amax(sigma['YY'])])
cross_pol_max = max([np.amax(sigma['XY']), np.amax(sigma['YX'])])
vmax = dict(zip(pols, [auto_pol_max, auto_pol_max, cross_pol_max, cross_pol_max]))
xticks = [N_freqs / 6 * l for l in range(6)]
xticks.append(N_freqs - 1)
xticklabels = [str(sigfig(freq_array[k]) * 10**(-6)) for k in xticks]
for k in range(4):
    sigma[pols[k]] = np.ma.masked_equal(sigma[pols[k]], 0)
    cmap = cm.cool
    cmap.set_bad(color='white')
    cax = temp_ax[k / 2][k % 2].imshow(sigma[pols[k]], cmap=cmap, vmin=0, vmax=vmax[pols[k]])
    cbar = temp_fig.colorbar(cax, ax=temp_ax[k / 2][k % 2])
    cbar.set_label('Sigma (~Temperature)')
    temp_ax[k / 2][k % 2].set_title(pols[k])
    temp_ax[k / 2][k % 2].set_ylabel('Observation')
    temp_ax[k / 2][k % 2].set_xlabel('Frequency (Mhz)')
    temp_ax[k / 2][k % 2].set_xticks(xticks)
    temp_ax[k / 2][k % 2].set_xticklabels(xticklabels)
    temp_ax[k / 2][k % 2].set_aspect(float(sigma[pols[k]].shape[1]) / sigma[pols[k]].shape[0])
    temp_ax[k / 2][k % 2].xaxis.set_minor_locator(AutoMinorLocator(4))

if curve_calc:
    curves = {}
    for pol in pols:
        for f in curve_freqs:
            curves[pol] = sigma[pol][:, f]
    cu_fig, cu_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    cu_fig.suptitle('HERA Golden Set Sigmas, for Interesting Frequencies, ' + cal_title)
    if LST:
        LST_array = np.load('/Users/mike_e_dubs/HERA/GS_LST.npy')
    for k in range(4):
        cu_ax[k / 2][k % 2].set_title(pols[k])
        cu_ax[k / 2][k % 2].set_ylabel('Sigma')
        for f in curve_freqs:
            cu_ax[k / 2][k % 2].plot(range(len(sigma[pols[k]][:, f])),
                                     sigma[pols[k]][:, f],
                                     label=str(sigfig(freq_array[f]) * 10**(-6)))
        cu_ax[k / 2][k % 2].legend()
        if LST:
            xticklabels = [str(sigfig(LST_array[int(tick)])) for tick in
                           cu_ax[k / 2][k % 2].get_xticks()[1:-1]]
            xticklabels.insert(0, '0')
            xticklabels.append('0')
            cu_ax[k / 2][k % 2].set_xticklabels(xticklabels)
            cu_ax[k / 2][k % 2].set_xlabel('LST (hours)')
        else:
            cu_ax[k / 2][k % 2].set_xlabel('Observation')

plt.tight_layout()

hist_fig.savefig(outpath + hist_fig_name)
temp_fig.savefig(outpath + temp_fig_name)
cu_fig.savefig(outpath + cu_fig_name)
