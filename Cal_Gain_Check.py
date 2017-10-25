import pyuvdata
import glob
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from matplotlib import cm

inpath = '/data6/HERA/data/2458042/'
first_pathlist = glob.glob(inpath + '*.first.calfits')
omni_pathlist = glob.glob(inpath + '*.omni.calfits')
first_outpath = '/data4/mwilensky/cal_gains/first/'
omni_outpath = '/data4/mwilensky/cal_gains/omni/'
pol_titles = {-1: 'rr', -2: 'll', -3: 'rl', -4: 'lr', -5: 'xx', -6: 'yy', -7: 'xy', -8: 'yx'}

first_cal = pyuvdata.UVCal()
omni_cal = pyuvdata.UVCal()

for path in first_pathlist:
    first_cal.read_calfits(path)
    end = path.find('.HH')
    obs = path[path.find('zen'):end + 3]
    pol = path[end - 2:end]
    gain_amp = np.absolute(first_cal.gain_array)

    dim = ceil(np.sqrt(first_cal.Nants_data))
    fig, ax = plt.subplots(figsize=(14, 8), nrows=dim, ncols=dim)
    fig.suptitle('First Cal Gain Amplitude ' + pol)
    cmap = cm.plasma
    vmin = np.amin(gain_amp)
    vmax = np.amax(gain_amp)

    for ant_num in range(first_cal.Nants_data):
        y = ant_num / dim
        x = ant_num % dim
        cax = ax[y][x].imshow(gain_amp[ant_num, 0, :, :, 0], cmap=cmap, vmin=vmin, vmax=vmax)
        if x > 0:
            ax[y][x].set_yticks([])
        else:
            ax[y][x].set_ylabel('Frequency (Channel #)')
        if y < dim - 1:
            ax[y][x].set_xticks([])

    cbar = fig.colorbar(cax)
    plt.tight_layout()
    fig.savefig(first_outpath + obs + '_first_cal_gain.png')

for path in omni_pathlist:
    omni_cal.read_calfits(path)
    end = path.find('.HH')
    obs = path[path.find('zen'):end + 3]
    gain_amp = np.absolute(first_cal.gain_array)

    dim = ceil(np.sqrt(first_cal.Nants_data))

    for p in range(omni_cal.Njones):
        pol = pol_titles[omni_cal.jones_array[p]]
        fig, ax = plt.subplots(figsize=(14, 8), nrows=dim, ncols=dim)
        fig.suptitle('First Cal Gain Amplitude ' + pol)
        cmap = cm.plasma
        vmin = np.amin(gain_amp)
        vmax = np.amax(gain_amp)

        for ant_num in range(first_cal.Nants_data):
            y = ant_num / dim
            x = ant_num % dim
            cax = ax[y][x].imshow(gain_amp[ant_num, 0, :, :, p], cmap=cmap, vmin=vmin, vmax=vmax)
            if x > 0:
                ax[y][x].set_yticks([])
            else:
                ax[y][x].set_ylabel('Frequency (Channel #)')
            if y < dim - 1:
                ax[y][x].set_xticks([])
            else:
                ax[y][x].set_xlabel('Time (10s)')

        cbar = fig.colorbar(cax)
        plt.tight_layout()
        fig.savefig(first_outpath + obs + pol + '_first_cal_gain.png')
