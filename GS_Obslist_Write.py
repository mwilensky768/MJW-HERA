import glob
import numpy as np

inpath = '/Users/mike_e_dubs/HERA/Temperatures/GS_Calibrated_Hists/'

pathlist = glob.glob(inpath + '*xx*bins.npy')
obs = [int(path[path.find('2458042') + 8:path.find('.xx')]) for path in pathlist]

np.savetxt('/Users/mike_e_dubs/HERA/Temperatures/GS_obs.txt', obs)
