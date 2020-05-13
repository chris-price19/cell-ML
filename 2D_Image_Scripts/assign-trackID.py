#!/usr/bin/python

"""
Version 3/19/2020

Import and visualize data from the 3D cell microscopy experiments. 

"""

import numpy as np
import pandas as pd
import os
import sys
import re

import matplotlib.pyplot as plt
from scipy import stats
from cell_data_loader import uniq, cell_dataframe

def locate_trackID(df, xmatch, ymatch, tol = 25):

	subDF = df.groupby('trackID').mean()
	# print(subDF)

	tracklist = subDF.loc[(np.abs(subDF['x'].values - xmatch) <= tol) & (np.abs(subDF['y'].values - ymatch) <= tol)]

	return tracklist.index.tolist()

print(pd.__version__)
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
cwd = os.getcwd()

if 'Chris Price' in cwd:
	datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
else:
	print('add your path to Plasticity_Protrusions_ML here')
	sys.exit()


##### in this section, load a dataframe which contains the x-y coordinates of the images to be labeled with a track ID.

# expdata = pd.read_excel(datadir+"\\research\\cell ML\\modified_cell_protrusion_v1.xlsx", skiprows=12)
matchdata = pd.read_excel(boxdir+"\\research\\cell ML\\modified_cell_protrusion_v1.xlsx", skiprows=0, nrows=10)

######


#### clean up the data to be matched

cols=[i for i in matchdata.columns if i not in ["index","blank"]]
# print(cols)
for col in cols:
    matchdata[col] = pd.to_numeric(matchdata[col], errors='coerce')

print(expdata)
print(matchdata)
# print(np.nanmax(expdata.iloc[:,1:]))
# print(np.nanmin(expdata.iloc[:,1:]))
# sys.exit()

############
###########

##### get ready to load the tracking data

key = pd.read_excel(datadir +'\\images\\track-data\\computational_key.xlsx')
dir1 = datadir + '\\combined_data'

tol = 30

fig, ax = plt.subplots(1,8,figsize=(32, 5))
pcount = 0

dfcol = 'area'

# loop over each cell to be matched 

for ci, cn in enumerate(cols): 

	# get filters from the image for which part of track data to look at
	if ~np.isnan(matchdata[cn].iloc[0]):

		components = cn.split('_')
		assay = int(components[0].split('Q')[1])
		sample = components[1]
		plastic = components[2]
		cell = components[3]

		## interpolate if assay > 30 #### !!! ### can probably delete
		if assay > 30:

			components = cn.split('_')
			assay = int(components[0].split('Q')[1])
			sample = '_'.join(components[1:-2])
			plastic = components[-2]
			cell = components[-1]

			first = expdata[cn].first_valid_index()
			last = expdata[cn].last_valid_index()+1
			expdata[cn].iloc[first:last] = expdata[cn].iloc[first:last].interpolate(method='linear')


		### load the relevant track data.

		subkey = key.loc[key['assay'] == assay]
		cellDF = cell_dataframe(subkey, [dir1])
		cellDF = cellDF.loc[cellDF['plastic'] == plastic].loc[cellDF['sample'] == sample]

		# x and y coordinates of cell center in image (in units of micrometers)
		## important: need another script to correlate pixel location with distance.
		## very important: need to vertically reflect the images.

		xmatch = matchdata[cn].iloc[5]
		ymatch = matchdata[cn].iloc[6]


		## this section used for aligning in time. can probably be cut if using average x-y. the second to last frame of the image aligns with the last time of the longest cell in the track data. Can rework this to run backwards.

		maxlength = np.amax(cellDF.groupby('trackID').count().values)
		alltimes = pd.DataFrame(pd.unique(cellDF['time']), columns=['imaris_time'])
		
		sub_match = matchdata.filter(regex=('_'.join(cn.split('_')[0:3]) + '_' +'.*'))
		maxexplength = int(sub_match.iloc[7,1])

		candidates = []
		while len(candidates) < 1:
				candidates = locate_trackID(cellDF, xmatch, ymatch, tol = tol)
				tol += 5

		# candidates = locate_trackID(cellDF, xmatch, ymatch, tol = tol)

		offset_dex = maxexplength - maxlength

		corrscore = []
		lengths = []
		outDF = pd.DataFrame()

		for ci, cc in enumerate(candidates):

			subDF = cellDF.loc[cellDF['trackID'] == cc] #.reset_index(drop=True)
			# maxlength = np.amax(cellDF.groupby('trackID').count().values)

			# print(subDF)

			exp = expdata[['Original Time',cn]].iloc[offset_dex:].reset_index(drop=True)
			exp = pd.concat((alltimes,exp), axis=1)
			exp.columns = ['imaris_time','exp_time','exp_length_um']

			subDF = subDF.merge(exp, left_on = 'time', right_on='imaris_time',how='inner')
		
			if len(subDF) > 1: 

				
				# plotting
				# ax[pcount].scatter(subDF['exp_length_um'], subDF[dfcol])
				# ax[pcount].set_ylim(arealimits)

				slope, intercept, r_value, p_value, std_err = stats.linregress(subDF.dropna(subset=['exp_length_um'])['exp_length_um'], subDF.dropna(subset=['exp_length_um'])[dfcol])

			else:
				r_value = 0.

			corrscore.append(r_value**2)
			lengths.append(len(subDF))

			if corrscore[-1] == np.amax(corrscore):
				saveDF = subDF.copy()

		outDF = pd.concat((outDF, saveDF), axis=0)

		pcount += 1
		print(corrscore)
		print(lengths)

		matchdata.at[8, cn] = candidates[np.argmax(corrscore)]
		matchdata.at[9, cn] = offset_dex


# plt.show()

print(matchdata)