import pyuvdata

UV = pyuvdata.UVData()

UV.read_miriad('/Users/mike_e_dubs/python_stuff/miriad/temp_HERA_data/zen.2457555.40356.xx.HH.uvc')


UV.write_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/zen.2457555.40356.xx.HH.uvfits',
                force_phase=True, spoof_nonessential=True)
