import pyuvdata
import glob
import numpy as np
from math import pi

inpath = '/data6/HERA/data/2458042/zen.2458042.'
pathlist = glob.glob(inpath + '*.xx*.uv')
obslist = np.sort(np.array([int(path[path.find('zen.') + 12:path.find('.xx')])
                    for path in pathlist]))

pathlist_sort = [inpath + str(obs) + '.xx.HH.uv' for obs in obslist]

UV = pyuvdata.UVData()
LST = []

for path in pathlist_sort:
    UV.read_miriad(path)
    LST.append(UV.lst_array[0])

np.save('/data4/mwilensky/GS_LST.npy', np.array(LST) * 23.934 / (2 * pi))
