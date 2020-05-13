#!/usr/bin/python

"""
Version 4/14/2020

Split .tifs of cells into individual frames with time index labels. veritcally reflect each image.

"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os
import sys
import re
import timeit

import PIL.Image as pimage

def uniq(input):
	output = []
	for x in input:
		if x not in output:
			output.append(x)
	return output

print(pd.__version__)
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
cwd = os.getcwd()

if 'Chris Price' in cwd:
	datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
else:
	print('add your path to Plasticity_Protrusions_ML here')
	sys.exit()

os.chdir(datadir +'\\images\\raw-images\\')

files = os.listdir()

for fi, ff in enumerate(files):

	print(ff)

	filebase = ff.split('.')[0]

	image = pimage.open(ff)
	
	for ii in np.arange(160):
		try:
			image.seek(ii)
		except:
			maxtime = ii
			break

	os.chdir(datadir + '\\images\\split-images')
	# print(maxtime)
	for ii in np.arange(maxtime):

		fname = filebase + '_' + str(ii+1) +'of' + str(maxtime) + '.png'
		image.seek(ii)

		# flip image vertically, now bottom left corner of the image corresponds to origin of position in track-data.
		frame = image.copy().transpose(pimage.FLIP_TOP_BOTTOM).convert('L')
	
		frame.save(fname)
	
	os.chdir(datadir +'\\images\\raw-images\\')
		# plt.imshow(frame)
		# print(frame)
		# plt.show()
		