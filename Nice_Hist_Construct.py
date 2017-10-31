import rfipy
import matplotlib.pyplot as plt

obspath = '/data6/HERA/data/2458042/zen.2458042.31939.xx.HH.uvOR'
obs = 'zen.2458042.31939.xx.HH'

RFI = rfipy.RFI(obs, obspath, bad_time_indices=[], filetype='miriad',
                good_freq_indices=range(64, 960))

data = RFI.one_d_hist_prepare(fit=True, bin_window=[10**(-7.5), 10**(-1)])

fig, ax = plt.subplots(figsize=(14, 8))
title = obs + ' Visibility Difference Histogram'

RFI.one_d_hist_plot(fig, ax, data, title)

fig.savefig('/data4/mwilensky/test_obs.png')
