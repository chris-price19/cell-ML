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

def uniq(input):
	output = []
	for x in input:
		if x not in output:
			output.append(x)
	return output

def cell_dataframe(key, dirs):

	cwd = os.getcwd()

	masterDF = pd.DataFrame()

	for dd in dirs:
		print(dd)
		os.chdir(dd)

		for ai, aa in enumerate(uniq(key['assay'])):
			
			adir = 'QuantInv'+str(aa)
			os.chdir(adir)
			print(adir)
			samplelist = uniq(key.loc[key['assay'] == aa,'sample'])

			for si, ss in enumerate(samplelist):

				os.chdir('Q'+str(aa)+'_'+ss+'_Statistics')
				subkey = key.loc[key['assay'] == aa].loc[key['sample'] == ss]

				files = os.listdir()
				# print(files)
				fileprefix = files[int(np.floor(len(files)/2))].split(ss)[0] + ss
				# print(fileprefix + ss)

				if subkey['env'].values[0].rstrip() in files[0]:
					# print(key.loc[key['assay'] == aa].loc[key['sample'] == ss]['env'].values[0])
					fileprefix = files[0].split(ss)[0] + ss +'_' + subkey['env'].values[0].rstrip()

				# print(fileprefix)
				
				positions = pd.read_csv(fileprefix + '_Position.csv', skiprows = 3)
				# print(positions)
				positions = positions.loc[:, ['Position X', 'Position Y', 'Position Z', 'Time', 'TrackID','ID']]
				# print(positions['Time']*subkey['time_int'].values[0])
				positions['Time'] = pd.to_timedelta(positions['Time']*subkey['time_int'].values[0], unit='m')
				# sys.exit()
				
				###### sphericity
				sphericity = pd.read_csv(fileprefix + '_Sphericity.csv', skiprows = 3)
				# sphericity = sphericity.loc[:,['Sphericity','Time','TrackID','ID']]
				sphericity = sphericity.loc[:,['Sphericity','ID']]
				# cellDF = positions.merge(sphericity,how='left',on=['Time','TrackID'])
				cellDF = positions.merge(sphericity, how='inner', on=['ID'])

				##### oblate ellipticity
				elliptic_O = pd.read_csv(fileprefix + '_Ellipticity_(oblate).csv', skiprows = 3)
				# elliptic_O = elliptic_O.loc[:,['Ellipticity (oblate)','Time','TrackID','ID']]
				elliptic_O = elliptic_O.loc[:,['Ellipticity (oblate)','ID']]
				cellDF = cellDF.merge(elliptic_O, how='inner', on=['ID'])

				###### prolate ellipticity
				elliptic_P = pd.read_csv(fileprefix + '_Ellipticity_(prolate).csv', skiprows = 3)
				# elliptic_P = elliptic_P.loc[:,['Ellipticity (prolate)','Time','TrackID','ID']]
				elliptic_P = elliptic_P.loc[:,['Ellipticity (prolate)','ID']]
				cellDF = cellDF.merge(elliptic_P, how='inner', on=['ID'])

				#####

				###### area
				area = pd.read_csv(fileprefix + '_Area.csv', skiprows = 3)
				area = area.loc[:,['Area','ID']]
				cellDF = cellDF.merge(area, how='inner', on=['ID'])

				#####

				###### volume
				volume = pd.read_csv(fileprefix + '_Volume.csv', skiprows = 3)
				volume = volume.loc[:,['Volume','ID']]
				cellDF = cellDF.merge(volume, how='inner', on=['ID'])

				#####

				idDF = key.loc[key['assay'] == aa].loc[key['sample'] == ss]
				# print(len(cellDF))

				test = pd.concat([idDF]*len(cellDF)).reset_index(drop=True)
				# print(test)

				cellDF = pd.concat([cellDF, test], axis=1)

				## clean up cell ID algorithm here

				def clean_cellDF(cellDF, newcols):

					cellDF = cellDF.loc[cellDF['TrackID'].notna()].copy()
					cellDF.columns = newcols
					# trackIDs = uniq(cellDF['trackID'])
					# print(cellDF.head(35))
					cellDF['trackID'] = cellDF['trackID'] - 1e9
					# cellDF['trackID'].apply(int())
					cellDF = cellDF.set_index('time')
					# print(cellDF.index)
					# print(cellDF.sort_values(by=['trackID','time']).head(35))
					numeric = cellDF.select_dtypes('number').columns
					non_num = cellDF.columns.difference(numeric)
					d = {**{x: 'mean' for x in numeric}, **{x: 'first' for x in non_num}}
					# df.resample('10T').agg(d)

					cellDF = cellDF.groupby(by=['trackID']).resample('10T').agg(d)
					# print(cellDF.head(35))
					# sys.exit()
					# print(cellDF.index)

					cellDF[['x','y','z','spheric','ellipticO','ellipticP','area', 'volume']] = cellDF[['x','y','z','spheric','ellipticO','ellipticP','area','volume']].interpolate()
					cellDF[['trackID','assay','time_int','plastic','env','sample']] = cellDF[['trackID','assay','time_int','plastic','env','sample']].fillna(method='pad')
					# print(cellDF.head(35))
					# leave ID column inserted as NAN for now since there is no real reason to fill these / everything else is a random integer.

					## is it worth merging track IDs. probably not, at least for now. but that would go here.

					# print(cellDF.head(35))
					cellDF = cellDF.reset_index(level='trackID',drop=True)
					cellDF = cellDF.reset_index(level='time')
					# print(cellDF.head(35))
					
					return cellDF

				cellDF = clean_cellDF(cellDF, ['x','y','z','time','trackID','ID','spheric','ellipticO','ellipticP','area', 'volume', 'assay','sample','plastic','env','time_int'])

				## add relative time
				cellDF = cellDF.merge(cellDF[['assay','sample','trackID','time']].groupby(['assay','sample','trackID']).min(), how = 'left', on=['assay','sample','trackID'], suffixes=('','_min'))
				cellDF.loc[:,'rel_time'] = cellDF['time'] - cellDF['time_min']

				masterDF = masterDF.append(cellDF)

				os.chdir('..')
		
			os.chdir('..')

		os.chdir(cwd)
	
	# print('test')
	print(masterDF)

	return masterDF.reset_index(drop=True)

if __name__ == "__main":

	print(pd.__version__)
	pd.set_option('display.expand_frame_repr', False)
	cwd = os.getcwd()
	key = pd.read_excel(cwd+'/computational_key.xlsx')
	boxdir = "\\".join(cwd.split("\\")[:-4])
	# print(boxdir)

	# print(key)

	dir1 = boxdir + '\\Plasticity_Protrusions_ML\\combined_data'
	# dir2 = boxdir + '\\Penn\\Spring 2020\\ENM 531\\project\\HP_MP_LP_control_GS'

	keytest = key.loc[key['assay'] == 43]	

	readfile = 1

	if readfile == 1:
		dateparser = lambda x: pd.to_timedelta(x)   # datetime.strptime(x, '%H:%M:%S')
		cellDF = pd.read_csv('cellDF_cleaned.csv', parse_dates=['time'], date_parser=dateparser)
		# cellDF = pd.read_csv('cellDF_cleaned.csv', parse_dates=['time'])
		# dtype={'x':np.float64,'y':np.float64,'z':np.float64,'time':str,'trackID':np.int32,'ID':np.float64,'spheric':np.float64,'ellipticO':np.float64,'ellipticP':np.float64,'assay':np.int32,'sample':str,'plastic':str,'env':str,'time_int':np.float64}
		
		print(cellDF.dtypes)
		cellDF['time'] = pd.to_timedelta(cellDF['time'])

	else:
		cellDF = cell_dataframe(key, [dir1])
		cellDF.to_csv('cellDF_cleaned.csv', date_format = '%H:%M:%S')

	# print(cellDF)
	# print(cellDF.dtypes)
	# print('time test')
	# print(cellDF.loc[cellDF['Time_x'] != cellDF['Time_y']])
	# print('trackID test')
	# print(cellDF.loc[cellDF['TrackID_x'] != cellDF['TrackID_y']])

	### net distance traveled
	mins = cellDF.loc[cellDF.groupby(by=['assay','sample','trackID']).time.idxmin()]
	mins = mins.reset_index(drop=True)
	cellDF = cellDF.merge(mins[['x','y','z','assay','sample','trackID']], on=['assay','sample','trackID'], how='inner', suffixes=('','_min'))


	print(cellDF.head(10))
	# print(cellDF['time'] - pd.Timedelta(minutes=10))

	rightDF = cellDF.copy()
	rightDF['time'] = rightDF['time'] - pd.Timedelta(minutes=10)

	distanceDF = cellDF.merge(rightDF[['x','y','z','assay','sample','trackID','time']], how = 'left', on=['assay','sample','trackID', 'time'], suffixes=('','_nxt'))
	distanceDF = distanceDF.loc[distanceDF['x_nxt'].notna()].reset_index(drop=True)

	distanceDF = distanceDF.merge(distanceDF[['assay','sample','trackID','time']].groupby(['assay','sample','trackID']).min(), how = 'left', on=['assay','sample','trackID'], suffixes=('','_min'))

	# print(distanceDF['x_nxt'].isna().sum())
	# print(distanceDF.head(150))

	# distancedelta = np.sqrt(np.sum((distanceDF[['x_nxt','y_nxt','z_nxt']].values - distanceDF[['x','y','z']].values)**2,axis=1))
	distanceDF.loc[:,'net_d'] = np.sqrt(np.sum((distanceDF[['x','y','z']].values - distanceDF[['x_min','y_min','z_min']].values)**2,axis=1))
	distanceDF.loc[:,'stepsize'] = np.sqrt(np.sum((distanceDF[['x_nxt','y_nxt','z_nxt']].values - distanceDF[['x','y','z']].values)**2,axis=1))
	distanceDF.loc[:,'rel_time'] = distanceDF['time'] - distanceDF['time_min']
	print(distanceDF.head(10))



	# ##################################
	# groups = distanceDF.loc[distanceDF['plastic'] == 'M'].loc[distanceDF['assay'] == 29].groupby(by=['assay','sample','trackID'])
	# # groups = distanceDF.groupby(by=['assay','sample','trackID'])
	# for name, group in groups:
	# 	if len(group) == 60:
	# 		print(group)
	# 		plt.plot(np.arange(len(group)),group['spheric'])
	# 		plt.plot(np.arange(len(group)),group['ellipticO'])
	# 		plt.plot(np.arange(len(group)),group['ellipticP'])
	# 		plt.show()
	# 	print(len(group))
	# ##################################

	#################################################
	# ################# data inspection
	# fig, ax = plt.subplots(1,2,figsize=(10, 5))
	# # ax = fig.add_subplot(121)

	# ax[0].hist(distanceDF['stepsize'], bins=40 )
	# ax[0].set_title('step size')

	# for name, group in groups:
	# 	# print(group)
	# 	ax[1].plot(np.arange(len(group)), group['stepsize'].cumsum(), color = 'b')

	# fig1, ax1 = plt.subplots(1,2,figsize=(10, 5))

	# ax1[0].hist(distanceDF['net_d'], bins=20)
	# ax1[0].set_title('displacement from start')

	# for name, group in groups:
	# 	# print(group)
	# 	ax1[1].plot(np.arange(len(group)), group['net_d'], color = 'r')


	# fig2, ax2 = plt.subplots(1,2,figsize=(10, 5))

	# ax2[0].hist(distanceDF['ellipticO'], bins=20)
	# ax2[0].set_title('shape measure')


	# for name, group in groups:
	# 	# print(group)
	# 	ax2[1].scatter(np.arange(len(group)), group['spheric'], color = 'g')


	# fig3, ax3 = plt.subplots(1,3,figsize=(15, 5))

	# stepgroups = distanceDF.loc[distanceDF['env'] == 'DMSO'].groupby(by=['plastic'])
	# c=['b','r','g']
	# gi = 0

	# N = 500
	# alpha, beta = 1.5, 1.5
	# pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

	# for name, group in stepgroups:
	# 	print(gi)
	# 	domain = np.linspace(np.amin(group['stepsize']), np.amax(group['stepsize']), N)
	# 	ax3[gi].hist(group['stepsize'], bins=40, color = c[gi], density=True)

	# 	a,b,m,s = pconv(*stats.levy_stable._fitstart(group['stepsize']))
	# 	print('%f %f %f %f' % (a,b,m,s))
		
	# 	# sys.exit()
	# 	# m, s = stats.levy_stable.fit(group['stepsize'])
	# 	pdf_levy = stats.levy_stable.pdf(domain, a, b, m, s)
	# 	ax3[gi].plot(domain, pdf_levy, color='k')

	# 	ax3[gi].set_title(name)
	# 	ax3[gi].set_xlim([-2,20])
	# 	gi+=1

	# plt.show()
	# #############################
	#######################################################################

	# #######################################################################
	# ## comparison of step size distribution for control vs. drugs for HP 

	# fig4, ax4 = plt.subplots(1,2,figsize=(10, 5))

	# control = distanceDF.loc[distanceDF['plastic'] == 'H'].loc[distanceDF['env'].isin(['DMSO','dH2O'])]
	# inhibit = distanceDF.loc[distanceDF['plastic'] == 'H'].loc[~distanceDF['env'].isin(['DMSO','dH2O'])]

	# print('control')
	# print(control.head(50))
	# print(uniq(control['env']))
	# print('inhibit')
	# print(uniq(inhibit['env']))
	# print(inhibit.head(50))

	# c=['b','r','g']
	# gi = 0

	# N = 500
	# alpha, beta = 1.5, 1.5
	# pconv = lambda alpha, beta, mu, sigma: (alpha, beta, mu - sigma * beta * np.tan(np.pi * alpha / 2.0), sigma)

	# # for name, group in stepgroups:
	# print(gi)
	# domain = np.linspace(np.amin(control['stepsize']), np.amax(control['stepsize']), N)
	# ax4[gi].hist(control['stepsize'], bins=40, color = c[gi], density=True)

	# a,b,m,s = pconv(*stats.levy_stable._fitstart(control['stepsize']))
	# print('%f %f %f %f' % (a,b,m,s))

	# # sys.exit()
	# # m, s = stats.levy_stable.fit(group['stepsize'])
	# pdf_levy = stats.levy_stable.pdf(domain, a, b, m, s)
	# ax4[gi].plot(domain, pdf_levy, color='k')

	# ax4[gi].set_title('control')
	# ax4[gi].set_xlim([-2,20])

	# gi += 1

	# domain = np.linspace(np.amin(inhibit['stepsize']), np.amax(inhibit['stepsize']), N)
	# ax4[gi].hist(inhibit['stepsize'], bins=40, color = c[gi], density=True)

	# a,b,m,s = pconv(*stats.levy_stable._fitstart(inhibit['stepsize']))
	# print('%f %f %f %f' % (a,b,m,s))

	# # sys.exit()
	# # m, s = stats.levy_stable.fit(group['stepsize'])
	# pdf_levy = stats.levy_stable.pdf(domain, a, b, m, s)
	# ax4[gi].plot(domain, pdf_levy, color='k')

	# ax4[gi].set_title('inhibit')
	# ax4[gi].set_xlim([-2,20])

	# plt.show()

	# #####################################################################

	#######################################################################

	# population average of elliptic trend group by plasticity (control only)

	plgroups = ['L','M','H']
	c = ['b','r','g']

	fig5, ax5 = plt.subplots(1,1,figsize=(10, 5))

	for pi, pp in enumerate(plgroups):

		controls = distanceDF.loc[distanceDF['plastic'] == pp].loc[~distanceDF['env'].isin(['DMSO','dH2O'])]
		# controls = distanceDF.loc[distanceDF['plastic'] == pp].loc[distanceDF['env'].isin(['GM6001'])]
		controls['net_d'] = controls['net_d'] ** 2

		timegroupmeans = controls[['net_d','stepsize','spheric','ellipticO','ellipticP','area', 'volume', 'rel_time']].groupby(by=['rel_time']).mean()
		timegroupstd = controls[['net_d','stepsize','spheric','ellipticO','ellipticP','area', 'volume','rel_time']].groupby(by=['rel_time']).std()

		# need to subtract from the times the minimum time for each group by assay, sample, trackID

		# print(timegroupmeans.index / np.timedelta64(1,'m'))
		quantity = 'area'

		# ax5.plot(timegroupmeans.index / np.timedelta64(1,'m'), timegroupmeans[quantity], color=c[pi], label=pp)
		# ax5.plot(timegroupmeans.index / np.timedelta64(1,'m'), timegroupmeans[quantity] + timegroupstd[quantity], color=c[pi], alpha=0.5)
		# ax5.plot(timegroupmeans.index / np.timedelta64(1,'m'), timegroupmeans[quantity] - timegroupstd[quantity], color=c[pi], alpha=0.5)

		ax5.scatter(controls['area'], controls['volume'], label='A-V corr')

		ax5.set_xlabel('time (min)')
		ax5.set_title(quantity)
		# ax5.set_ylim([0,1])


	figx, axx = plt.subplots(1,1)
	rgrid = np.linspace(0,50,200)
	axx.plot(np.pi*rgrid**2, 4/3*np.pi*rgrid**3)

	plt.legend()
	plt.show()

	# start = pd.Timestamp('20180606')
	# new_index = start + test.index
	# test.index = new_index 
	# index_plot = new_index[::20000] # to reduce the number of labels on x-axis 
	# labels = index_plot.strftime('%H:%M:%S') 
	# plt.plot(test) plt.xticks(index_plot, labels, rotation=60) plt.show()