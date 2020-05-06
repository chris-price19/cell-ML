#!/usr/bin/python

"""
Version 4/14/2020

Segment split .tifs and identify a cell ID for each cell taken from an assay / plasticity / condition.

"""


import numpy as np
import pandas as pd

import os
import sys
import re
import timeit

from shutil import copy2, rmtree
from zipfile import ZipFile

from PIL import ImageFile
from io import BytesIO

def uniq(input):
	output = []
	for x in input:
		if x not in output:
			output.append(x)
	return output


def clean_images(datadir):

	os.chdir(datadir +'images\\single-images\\')

	with ZipFile(datadir + 'images\\single-images\\indiv-cells.zip', 'r') as zipObj:
        	files = zipObj.namelist()

	key = pd.read_excel(datadir + 'images\\track-data\\computational_key.xlsx')

	# filter key to data currently using
	key = key.loc[key['assay'] <= 30]

	groups = key.groupby(['assay','plastic','env']).groups

	# set the required consecutive frames to keep
	min_lags = 6

	dcount = 0
	fcount = 0

	globalcell_ind = 0

	newstringlist = []

	# read and write directly from zip file to help cloud storage
	zipIn = ZipFile(open(datadir + 'images\\single-images\\indiv-cells.zip', 'rb'))
	zipOut = ZipFile(datadir +'images\\data-images\\cell_series.zip', 'w')

	for ai, aa in enumerate(groups):

		stringheader = 'Q'+str(aa[0]) + '_' + aa[1] + '_' + aa[2]
		print(stringheader)

		subfiles = sorted([f for f in files if stringheader in f])
		
		if len(subfiles) < 1:
			# skip empty groups without any segmented images
			continue	

		cells = sorted(uniq([int(f.split('_')[-1].split('.')[0]) for f in subfiles]))
		
		for ci, cc in enumerate(cells):

			if cc >= 500:
				# leave out cells which were assigned 'new' IDs in the segmentation
				continue

			series = [f for f in subfiles if int(f.split('_')[-1].split('.')[0]) == cc]
			series_times = sorted([int(f.split('_')[-2].split('of')[0]) for f in series])
			stringtail = series[0].split('of')[-1]

			# filter out for time IDs for this cell ID which are part of a consecutive sequence with length of at least min_lags
			seqs = [f for f in np.split(series_times, np.where(np.diff(series_times) != 1)[0]+1) if len(f) >= min_lags]
			# print(seqs)
			if len(seqs) < 1:
				continue

			for ti, tt in enumerate(seqs):

				# reset time to t = 0 for all cells.
				min_time = np.amin(tt)
				# numcells = len(tt)
				# cellid_map = np.concatenate((mastercell_idlist[prevcell_ind:prevcell_ind+numcells, None], np.array(cells)[:,None]), axis=1)

				for si, ss in enumerate(tt):

					oldstring = stringheader = 'Q'+str(aa[0]) + '_' + aa[1] + '_' + aa[2] + '_' + str(ss) + 'of' + stringtail
					# print(oldstring)
					currenttime = ss - min_time
					
					# globalcell_ind gives each consecutive track of at least min_lags a unique ID.
					newstring = 'c' + str(globalcell_ind) + '_t' + str(currenttime) + '_' + aa[1] + '_' + aa[2] +'.png'

					if newstring in newstringlist:
						print('duplicate')
						print(newstring)
						print(ci)
						sys.exit()
					else:
						newstringlist.append(newstring)

					dcount += 1

					zipOut.writestr(newstring, zipIn.read(oldstring))
					
				globalcell_ind += 1

	zipOut.close()
	zipIn.close()

	print(dcount)
	print(fcount)

	return

def lag_filter(min_lags):



	return


def zip_directory(direc, name):
	
	cwd =os.getcwd()
	os.chdir(direc)

	files = os.listdir()
	print(len(files))
	files = [f for f in files if '.zip' not in f]
	print(len(files))

	# zipOut = ZipFile(name +'.zip', 'w')
	# for fi, ff in enumerate(files):

	# 	with open(ff, 'rb') as file:

	# 		zipOut.writestr(ff, file.read())

	# zipOut.close()

	for fi, ff in enumerate(files):
		os.remove(ff)
	
	os.chdir(cwd)

	return

print(pd.__version__)
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
cwd = os.getcwd()

if 'Chris Price' in cwd:
	datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
else:
	print('add your path to Plasticity_Protrusions_ML here')
	sys.exit()

clean_images(datadir)

# zip_directory('C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\images\\single-images\\', 'segmented_with_imaris_tracks')
##############################
