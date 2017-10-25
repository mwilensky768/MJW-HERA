import pyuvdata as pyuv
import csv
import numpy as np

UV = pyuv.UVData()

UV.read_miriad('/Users/mike_e_dubs/python_stuff/miriad/temp_HERA_data/zen.2457555.40356.xx.HH.uvc')

keys = ['Nants_telescope', 'Nants_data', 'Nbls', 'Nblts', 'Nfreqs', 'Npols', 'Ntimes',
        'channel_width', 'vis_units']

values = []

for key in keys:
    values.append(getattr(UV, key))

properties = dict(zip(keys, values))

w = csv.writer(open('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/2457555.40356.xx.props.csv', 'w'))

for key, val in properties.items():
    w.writerow([key, val])

unique_times = np.unique(UV.time_array)
print(unique_times[1] - unique_times[0])
