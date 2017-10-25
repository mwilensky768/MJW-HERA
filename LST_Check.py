import pyuvdata
import glob
import numpy as np
from math import pi

pathlist = 3 * ['/data6/HERA/data/2458042/zen.2458042.']
obs = ['12552','48343','53563']
for k in range(3):
    pathlist[k] += obs[k] + '.xx.HH.uv'

UV0 = pyuvdata.UVData()
UV48 = pyuvdata.UVData()
UV55 = pyuvdata.UVData()

UV0.read_miriad(pathlist[0])
UV48.read_miriad(pathlist[1])
UV55.read_miriad(pathlist[2])

begin = UV0.lst_array[0]

ev_beg = UV48.lst_array[0] + 2 * pi
end = UV55.lst_array[0] + 2 * pi

lst = np.array([begin, ev_beg, end]) * 24 / (2 * pi)
lst_len = np.diff(lst)

print('The LSTs of the events were: ' + str(lst))
print('The differences between the events were ' + str(lst_len))
