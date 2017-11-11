import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt
import glob

outpath = '/data4/mwilensky/catalogs/2458056/uncalibrated/'
writepath = '/data4/mwilensky/temperatures/2458056/uncalibrated/'
flag_slices = ['Unflagged', 'All']
pathlist = glob.glob('/data6/HERA/data/2458042/*.uv')
catalog_type = 'waterfall'
plot_type = 'freq-time'
band = {'Unflagged': 'fit', 'All': [1e-0.5, 1e+4]}
fit = {'Unflagged': True, 'All': False}
fit_window = [0, 1e+012]
bin_window = [1e-7.5, 1]
bad_time_indices = []
auto_remove = True
good_freq_indices = range(64, 960)
bins = np.logspace(-7.5, 3, num=1001)
write = {'Unflagged': True, 'All': False}
ant_pol_times = range(55)
ant_pol_freqs = [316, 317, 318, 319, 320, 321, 322, 406, 787, 788, 849, 869, 870]

for path in pathlist:
    start = path.find('zen.')
    end = path.find('.uv')
    obs = path[start:end]

    RFI = rfi.RFI(obs, path, filetype='miriad', bad_time_indices=bad_time_indices,
                  auto_remove=auto_remove, good_freq_indices=good_freq_indices)

    if catalog_type is 'waterfall':

        RFI.rfi_catalog(outpath, band=band, flag_slices=flag_slices, fit=fit,
                        fit_window=fit_window, bin_window=bin_window,
                        write=write, plot_type=plot_type, writepath=writepath)

    elif catalog_type is 'temperature':

        RFI.one_d_hist_prepare(flag_slice=flag_slices[0], bins=bins, fit=fit,
                               bin_window=bin_window, fit_window=fit_window,
                               temp_write=temp_write, write=write,
                               writepath=outpath)

    elif plot_type is 'ant-pol':

        RFI.ant_pol_catalog(outpath, ant_pol_times, ant_pol_freqs)
