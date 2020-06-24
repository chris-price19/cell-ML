# cell-ML
Image classification of cell environments ENM 531

# Data Structure
The parent folder contains several subfolders of the image statistics along with a folder dedicated just to images. All I/O code should reference the parent folder as the 'root' to make it portable for anyone with access to the parent folder.

 - images/  
      - raw-images/: contains .tif files labeled as: assay_plasticity_condition.tif  
      - split-images/: derived from raw-images. contains split .tif files labeled as assay_plasticity_condition_timeIndex.tif  
      - single-images/: derived from split-images. contains uniform-sized images of single cells labeled as assay_plasticity_condition_timeIndex_cellID.png
      - track-data/: contains the IMARIS processed data, transformed and cleaned
      - data-images/: contains cleaned and re-labeled single cell images with consecutive tracks, splitting same cell 'tracks' into unique IDs if there are independent consecutive sequences. relabels all time indexes to start at 0.

# TO-DO

1) Optimize hyperparameters / try to get successful training on CNN + LSTM with images
      - may require using pretrained CNN as pre-filter for LSTM

2) Gaussian Process for protrusion length?
      - may not be necessary for experimentalists
      
3) Can we target PLoS Comp Bio? What about further network, architecture optimization?
      - transformer instead of LSTM
      - improve image pre-processing (segmentation and tracking)

4) Adding additional feature columns to the 3D data
      - should be available soon, chris will follow up
      
5) Identifying drugs within a single plasticity using CNN + LSTM architecture?
      
6) Using occlusion to do feature importance with CNN  layers.
      - can do this in both the image and vectorized data
