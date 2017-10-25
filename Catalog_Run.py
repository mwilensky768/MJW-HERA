import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt
import glob

outpath = '/data4/mwilensky/temperatures/golden_set/uncalibrated/'
plotlist = glob.glob(outpath + '*.png')
flag_slices = ['Unflagged', ]
N_plot = 1
if plotlist:
    proc_start = len(plotlist) / N_plot
    pathlist = glob.glob('/data6/HERA/data/2458042/*.uv')[proc_start:]
else:
    pathlist = glob.glob('/data6/HERA/data/2458042/*.uv')
catalog_type = 'temperature'
plot_type = 'freq-time'
band = 'fit'
fit = True
fit_window = [0, 10**12]
bin_window = [10**(-7.5), 10**(-1)]
bad_time_indices = []
auto_remove = True
good_freq_indices = range(64, 960)
bins = np.logspace(-7.5, 3, num=1001)
temp_write = True
write = True
ant_pol_times = range(55)
ant_pol_freqs = [316, 317, 318, 319, 320, 321, 322, 406, 787, 788, 849, 869, 870]

for path in pathlist:
    start = path.find('zen.')
    end = path.find('.uvOR')
    obs = path[start:end]

    RFI = rfi.RFI(obs, path, filetype='miriad', bad_time_indices=bad_time_indices,
                  auto_remove=auto_remove, good_freq_indices=good_freq_indices)

    if catalog_type is 'waterfall':

        RFI.rfi_catalog(outpath, band=band, flag_slices=flag_slices, fit=fit,
                        fit_window=fit_window, bin_window=bin_window, plot_type=plot_type)

    elif catalog_type is 'temperature':

        RFI.one_d_hist_prepare(flag_slice=flag_slices[0], bins=bins, fit=fit,
                               bin_window=bin_window, fit_window=fit_window,
                               temp_write=temp_write, write=write,
                               writepath=outpath)

    elif plot_type is 'ant-pol':

        RFI.ant_pol_catalog(outpath, ant_pol_times, ant_pol_freqs)
