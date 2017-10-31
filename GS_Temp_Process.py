import numpy as np
import glob

obslist_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_obs.txt'
temp_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_Temperature.txt'

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(temp_path) as g:
    lines = g.read().split("\n")[1:]  # Ignore first line, which just shows column headers

obslist = np.array(obslist).astype(int)
lines = np.array(lines)

temp_avg_cont = np.zeros(len(obslist), dtype=float)
temp_avg_rxr = np.zeros(len(obslist), dtype=float)
obs_count = np.zeros(len(obslist), dtype=int)

lines_obs = np.array([line[10:15] for line in lines]).astype(int)

good_lines_obs = lines_obs[np.logical_and(min(obslist) < lines_obs,
                                          lines_obs < max(obslist))]

good_lines = lines[np.logical_and(min(obslist) < lines_obs,
                                  lines_obs < max(obslist))]

good_lines_temps_cont = np.array([line[18:27] for line in good_lines]).astype(float)

good_lines_temps_rxr = np.array([line[30:39] for line in good_lines]).astype(float)

for k in range(len(obslist) - 1):
    obs_count[k] = len(good_lines_obs[np.logical_and(obslist[k] < good_lines_obs,
                                                     good_lines_obs < obslist[k + 1])])
obs_count[-1] = obs_count[-2]
m = 0

for k in range(len(obs_count)):
    obs_count[k]
    temp_avg_cont[k] = np.sum(good_lines_temps_cont[m:m + obs_count[k]]) / obs_count[k]
    temp_avg_rxr[k] = np.sum(good_lines_temps_rxr[m:m + obs_count[k]]) / obs_count[k]
    m += obs_count[k]

np.save('/Users/mike_e_dubs/HERA/Temperatures/GS_temp_avg_cont.npy', temp_avg_cont)
np.save('/Users/mike_e_dubs/HERA/Temperatures/GS_temp_avg_rxr.npy', temp_avg_rxr)

print('The average container temperature was ' + str(np.mean(temp_avg_cont)))
