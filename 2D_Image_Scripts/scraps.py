	# for ki, kk in enumerate(keep_region):

		# 	subimage = image[minr:maxr, minc:maxc].T # transpose to restore positional orientation.


				# print(pairwise_dist)
				# sys.exit()

				# meanadj = 0.3 - np.mean(subimage[region.image])
				# subimage[region.image] += meanadj



				########### try to split doubles #################### skip for now
				# if np.amax([(maxr-minr),(maxc-minc)]) >= 36:

				# 	# meanadj = 0.5 - np.mean(subimage[region.image])
				# 	# subimage[region.image] += meanadj
					
				# 	label_image2, thresh = identify_cells(subimage, 0.5, 1, 'local')
					
				# 	print('hi')
				# 	print(len(regionprops(label_image)))

				# 	image_label_overlay = label2rgb(label_image2, image=subimage, bg_label=0)
				# 	ax2.imshow(image_label_overlay)
				# 	# plt.show()
				#####################################################################
				

		# match_track = subid_pool.iloc[np.argmin(pairwise_dist)].loc['trackID']
		# print(centroids[inds[0],:])
		# print(exp_dist1[np.argmin(exp_dist1)])
		# print(subid_pool.iloc[np.argmin(exp_dist1)])

		# match_track = subid_pool.iloc[np.argmin(pairwise_dist)].loc['trackID']
		# print(centroids[inds[1],:])
		# print(exp_dist2[np.argmin(exp_dist2)])
		# print(subid_pool.iloc[np.argmin(exp_dist2)])


		

		# sys.exit()






		# plt.hist(np.array(propslist))
		# plt.show()
		# print(image)


class ClusterRandomSampler(Sampler):
    r"""Takes a dataset with cluster_indices property, cuts it into batch-sized chunks
    Drops the extra items, not fitting into exact batches
    Arguments:
        data_source (Dataset): a Dataset to sample from. Should have a cluster_indices property
        batch_size (int): a batch size that you would like to use later with Dataloader class
        shuffle (bool): whether to shuffle the data or not
    """

    def __init__(self, data_source, batch_size=None, shuffle=True):
        self.data_source = data_source
        if batch_size is not None:
            assert self.data_source.batch_sizes is None, "do not declare batch size in sampler " \
                                                         "if data source already got one"
            self.batch_sizes = [batch_size for _ in self.data_source.cluster_indices]
        else:
            self.batch_sizes = self.data_source.batch_sizes
        self.shuffle = shuffle

    def flatten_list(self, lst):
        return [item for sublist in lst for item in sublist]

    def __iter__(self):

        batch_lists = []
        for j, cluster_indices in enumerate(self.data_source.cluster_indices):
            batches = [
                cluster_indices[i:i + self.batch_sizes[j]] for i in range(0, len(cluster_indices), self.batch_sizes[j])
            ]
            # filter our the shorter batches
            batches = [_ for _ in batches if len(_) == self.batch_sizes[j]]
            if self.shuffle:
                random.shuffle(batches)
            batch_lists.append(batches)

            # flatten lists and shuffle the batches if necessary
        # this works on batch level
        lst = self.flatten_list(batch_lists)
        if self.shuffle:
            random.shuffle(lst)
        return iter(lst)

    def __len__(self):
        return len(self.data_source)