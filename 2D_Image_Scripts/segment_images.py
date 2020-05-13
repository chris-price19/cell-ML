#!/usr/bin/python

"""
Version 4/14/2020

Segment split .tifs and identify a cell ID for each cell taken from an assay / plasticity / condition.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

import os
import sys
import re
import timeit
import skimage.filters
import skimage.measure

from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_local
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from PIL import Image as pimage
from PIL import ImageFile
from io import BytesIO
from zipfile import ZipFile
# import cv2

def uniq(input):
	output = []
	for x in input:
		if x not in output:
			output.append(x)
	return output

def identify_cells(image, sigma, sq, algo):

	image = skimage.filters.gaussian(image, sigma=sigma) #, multichannel=False, preserve_range=True)

	# print(image.shape)

	if algo == 'bimodal':
	# apply threshold
		thresh = threshold_otsu(image)
		print('thresh bimodal')
		print(thresh)
		
	elif algo == 'local':
		bsize = 35 # int(np.amin(image.shape) / 2) + 1
		thresh = threshold_local(image, bsize)
		print('thresh local')
		print(thresh)
	# thresh = 0.07

	# remove artifacts connected to image border
	bw = closing(image > thresh, square(sq))
	cleared = clear_border(bw)

	label_image = label(cleared)

	return image, label_image, thresh

print(pd.__version__)
print(skimage.__version__)
pd.set_option('display.expand_frame_repr', False, 'display.max_columns', None)
cwd = os.getcwd()

if 'Chris Price' in cwd:
	datadir = 'C:\\Users\\Chris Price\\Box Sync\\Plasticity_Protrusions_ML\\'
else:
	print('add your path to Plasticity_Protrusions_ML here')
	sys.exit()

os.chdir(datadir +'images\\split-images\\')

files = os.listdir()

dateparser = lambda x: pd.to_timedelta(x)   # datetime.strptime(x, '%H:%M:%S')
cellDF = pd.read_csv(datadir + 'images\\track-data\\cellDF_cleaned_exp.csv', parse_dates=['time'], date_parser=dateparser)
cellDF['time'] = pd.to_timedelta(cellDF['time'])

key = pd.read_excel(datadir + 'images\\track-data\\computational_key.xlsx')

# filter key to data currently using
key = key.loc[key['assay'] <= 30]

# filter again for testing only
# key = key.loc[key['assay']  <= 22]

groups = key.groupby(['assay','plastic','env']).groups

px_over_um = 1.374 * 4 # recalibrate in the loops

# # print(files)

# zipOut = ZipFile(datadir +'images\\single-images\\indiv-cells.zip', 'w')

for ai, aa in enumerate(groups):
	
	stringheader = 'Q'+str(aa[0]) + '_' + aa[1] + '_' + aa[2]
	print(stringheader)

	subfiles = sorted([f for f in files if stringheader in f])
	
	time_ids = sorted([int(f.split('_')[-1].split('.')[0].split('of')[0]) for f in subfiles],reverse=True)
	max_time = np.amax([int(f.split('_')[-1].split('.')[0].split('of')[1]) for f in subfiles])

	idpool = cellDF.loc[cellDF['assay'] == aa[0]].loc[cellDF['plastic'] == aa[1]].loc[cellDF['env'] == aa[2]][['trackID','time','x','y']]
	exp_times = pd.unique(idpool['time'])
	
	exp_iter = len(exp_times)
	print(exp_iter)

	for si, ss in enumerate(time_ids):

		px_over_um = 1.374 * 4 # recalibrate in the loops

		if exp_iter < 1:
			break

		print('time index %d' % ss)

		new_id = 500

		fname = stringheader +'_' + str(ss) + 'of' + str(max_time) + '.png'
		
		image = np.asarray(pimage.open(fname)) / 255

		image = resize(image, (4096, 4096), order = 3) # bi-cubic interpolation

		if np.mean(image) <= 0.0001:
			print('blank frame')
			print(np.mean(image))
			# skip image if the mean pixel value is basically black. usually the last image in the series 
			continue;

		subid_pool = idpool.loc[idpool['time'] == exp_times[exp_iter-1]]
		# print(subid_pool.loc[subid_pool['trackID'] == 104])

		fig, ax = plt.subplots(figsize=(10, 6))

		image, label_image, thresh = identify_cells(image, 1., 7, 'bimodal')
		image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
		ax.imshow(image_label_overlay)
		# plt.show()

		# determine an offset to better match tracks.
		# calibrate px_over_um
		centroids = np.zeros((len(regionprops(label_image)),2))
		for ri, region in enumerate(regionprops(label_image)):

			if region.area >= 100*4*4:
				
				centroids[ri,:] = region.centroid
				centroids[ri,0] = 1024 * 4 - centroids[ri,0]

		centroids = centroids[np.sum(centroids,axis=1) > 0]
		# print(centroids.shape)
		pairwise = dist.squareform(dist.pdist(centroids, 'euclidean'))
		np.fill_diagonal(pairwise, 9999999.)
		nearest_neighbors = np.amin(pairwise,axis=1)
		inds = np.argsort(-nearest_neighbors)
		# print(inds[0:2])
		# iso_ind = np.argmax(nearest_neighbors)

		exp_dist1 = np.sqrt(np.sum((centroids[inds[0],[0,1]] / px_over_um - subid_pool[['y','x']]) ** 2, axis=1)).values


		exp_dist2 = np.sqrt(np.sum((centroids[inds[1],[0,1]] / px_over_um - subid_pool[['y','x']]) ** 2, axis=1)).values


		px_over_um = np.sqrt(np.sum((centroids[inds[0],:] - centroids[inds[1],:])**2)) / np.sqrt(np.sum((subid_pool.iloc[np.argmin(exp_dist2)].loc[['y','x']] - subid_pool.iloc[np.argmin(exp_dist1)].loc[['y','x']])**2))
		print('recalibrated px_over_um')
		print(px_over_um)
		# print(pairwise)
		# print(pairwise.shape)
		# print(pairwise == pairwise.T)
		# sys.exit()
		# index, # minr, #minc, # maxr, #maxc, centx, centy, area
		keep_region = np.zeros((len(regionprops(label_image)),10))

		regionlist = regionprops(label_image)
		regionlist = sorted(regionlist, key=lambda x: (x.centroid[0],(x.centroid[1])))

		for ri, region in enumerate(regionlist):

		    # take regions with large enough areas
			minr, minc, maxr, maxc = region.bbox
			subimage = image[minr:maxr, minc:maxc]
			# print(ri)
			# print(np.mean(subimage))

			if region.area >= 100*4*4: # and np.amax([(maxr-minr),(maxc-minc)]) <= 40*4:

				keep_region[ri,0] = ri+1
				keep_region[ri,1:5] = region.bbox
				keep_region[ri,5:7] = region.centroid
				keep_region[ri,7] = region.area
				# print(region.image)

				keep_region[ri,[1,3,5]] = 1024 * 4 - keep_region[ri,[1,3,5]]

				# print(keep_region[ri,[5,6]] / px_over_um)

				pairwise_dist = np.sqrt(np.sum((keep_region[ri,[5,6]] / px_over_um - subid_pool[['y','x']]) ** 2, axis=1)).values
				match_track = subid_pool.iloc[np.argmin(pairwise_dist)].loc['trackID']
				
				# print(pairwise_dist[np.argmin(pairwise_dist)])
				
				# print(match_track)

				# print(subid_pool.iloc[np.argmin(pairwise_dist)])
				if pairwise_dist[np.argmin(pairwise_dist)] < 50:
					keep_region[ri,8] = match_track
					keep_region[ri,9] = pairwise_dist[np.argmin(pairwise_dist)]
				else:
					keep_region[ri,8] = new_id
					new_id += 1

		keep_region = keep_region[keep_region[:,0] != 0,:]
		
		dupid_pool = subid_pool.copy()
		# handle duplicates
		while (len(uniq(keep_region[:,8])) != len(keep_region[:,8])): # or np.sum(np.isnan(keep_region[:,8]))>0:
			print('duplicates')

			dupid_pool = dupid_pool[~dupid_pool['trackID'].isin(keep_region[:,8])]
			u, c = np.unique(keep_region[:,8], return_counts = True)
			dups = u[c>1]
			print(dups)
			print(c[c>1])

			for di, dd in enumerate(dups):
				print('enter')
				keep_region[(keep_region[:,8] == dd) & (keep_region[:,9] != np.amin(keep_region[keep_region[:,8] == dd, 9])),-2:] = np.nan
				print(keep_region[np.isnan(keep_region[:,8])])

				inds = np.argwhere(np.isnan(keep_region[:,8]))
				print(inds)

				for ni, nn in enumerate(inds):

					pairwise_dist = np.sqrt(np.sum((keep_region[nn,[5,6]] / px_over_um - dupid_pool[['y','x']]) ** 2, axis=1)).values
					match_track = dupid_pool.iloc[np.argmin(pairwise_dist)].loc['trackID']

					if pairwise_dist[np.argmin(pairwise_dist)] < 50:
						print('new match')
						keep_region[nn,8] = match_track
						keep_region[nn,9] = pairwise_dist[np.argmin(pairwise_dist)]
					else:
						print('new id')
						keep_region[nn,8] = new_id
						new_id += 1

		# print(keep_region)
		######## PLOTTING ##########
		# fig2, ax2 = plt.subplots(figsize=(10, 6))
		# plt.ion()
		# plt.show()
		for ri, region in enumerate(regionlist):

			minr, minc, maxr, maxc = region.bbox
			subimage = image[minr:maxr, minc:maxc]

			if ri+1 in keep_region[:,0]:

				match_track = keep_region[keep_region[:,0] == ri+1,8]
				meanadj = 0.5 - np.mean(subimage[region.image])
				subimage[region.image] += meanadj
				subimage[~region.image] = 0


				# #### concatenate extra part of original image
				# if minc >= 1*4 and minr >= 1*4 and maxr <= image.shape[0] - 1*4 and maxc <= image.shape[1] - 1*4:

				# 	subimage = np.concatenate((image[minr-1*4:minr,minc:maxc], subimage, image[maxr:maxr+1*4, minc:maxc]), axis=0)
					
				# 	subimage = np.concatenate((image[minr-1*4:maxr+1*4,minc-1*4:minc], subimage, image[minr-1*4:maxr+1*4,maxc:maxc+1*4]), axis=1)
				# ###################

				subimage = skimage.filters.gaussian(subimage, sigma=2.)
				finalsize = 48

				if np.amax([maxc-minc, maxr-minr]) < finalsize * 4:
					imsize = finalsize
				else:
					imsize = int(np.amax([maxc-minc, maxr-minr]) / 4 + 16)
					

				# subimage = np.pad(subimage, ((int(np.floor(48*4 - subimage.shape[0])/2), int(np.floor(48*4 - subimage.shape[0])/2)),(int(np.floor((48*4 - subimage.shape[1])/2)), int(np.floor((48*4 - subimage.shape[1])/2)))), mode='constant', constant_values=np.amin(subimage))
				subimage = np.pad(subimage, ((int(np.floor(imsize*4 - subimage.shape[0])/2), int(np.floor(imsize*4 - subimage.shape[0])/2)),(int(np.floor((imsize*4 - subimage.shape[1])/2)), int(np.floor((imsize*4 - subimage.shape[1])/2)))), mode='constant') 

				# print(subimage)

				# ### dynamic plot
				# ax2.imshow(subimage, cmap='gray', vmin=0, vmax=1)
				# plt.draw()
				# plt.pause(4)
				if match_track == 98:
					eecolor = 'green'
				else:
					eecolor = 'red'

				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
		                                  fill=False, edgecolor=eecolor, linewidth=1)
				ax.add_patch(rect)
				######

				# print(subimage.shape)
				if subimage.shape[0] != finalsize*4 or subimage.shape[1] != finalsize*4:
					if subimage.shape[0] > finalsize*4 or subimage.shape[1] > finalsize*4:
						aa = True
					else:
						aa = False
					subimage = resize(subimage, (finalsize*4, finalsize*4), order = 3, anti_aliasing = aa, preserve_range = True)

				if subimage.shape[0] != subimage.shape[1]:
					print(subimage.shape)
					print('shape is bad!')
					sys.exit()


				# ######## WRITE #############
				# print('writing image %f x %f' % (subimage.shape[0], subimage.shape[1]))

				# os.chdir(datadir +'images\\single-images\\')

				# im = pimage.fromarray(subimage * 255).copy().convert('L')
				# savename = fname.split('.')[0] + '_'+str(int(match_track)) + '.png'
				# imfile = BytesIO()
				# im.save(imfile, 'png')
				# zipOut.writestr(savename, imfile.getvalue())

				# # im.save(savename)

				# os.chdir(datadir +'images\\split-images\\')
				# # sys.exit()
				# ########## WRITE ############
		plt.tight_layout()
		plt.show()
		exp_iter -= 1
		
zipOut.close()
		# plt.tight_layout()
		# plt.show()

