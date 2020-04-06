# cell-ML
Image classification of cell environments ENM 531

# Data Structure
The parent folder contains several subfolders of the image statistics along with a folder dedicated just to images. All I/O code should reference the parent folder as the 'root' to make it portable for anyone with access to the parent folder.

- images/  
      - raw-images/: contains .tif files labeled as: assay_plasticity_condition.tif
      - split-images/: derived from raw-images. contains split .tif files labeled as assay_plasticity_condition_timeIndex.tif
      - single-images/: derived from split-images. contains uniform-sized images of single cells labeled as assay_plasticity_condition_timeIndex_cellID.png

Comments:
  - The time index should correspond to 10 minute intervals. For assays with number <= 30, each frame is taken every 10 minutes, and for assays with number > 30, each frame represents a 20 minute interval. If necessary to use these, can probably linearly interpolate the images to get consistent time series data. But timeIndex should be 0 = 10min, 1 = 20min, etc.
  - Depending on how cell tracks are identified, eventually it would be good to map cellID to trackID.

# TO-DO

1) Segment images into square images of individual cells on black background.
  - First, split each .tif into individual frames.
  - Segment each frame to identify cells. Store the position of the cell, and / or use Kalman filter to connect cells across time slices.
    - Ideally, the position can be mapped to micrometers so that Imaris track ID can label each cell.
    - Check that the segmentation is effective, probably want to eliminate tracks with overlapping cells
  - Choose a square image size based on the largest cell identified across all images.
  - Crop cell pixels out of main .tif and add to black image of square size from above.
 
 2) Label the cells
    - Label categories: Environment plasticity: Low, Medium, High
                        Drug condition: DMSO / dH2O (control), GM6001, other drugs listed in the key
 
 3) Use convolutional neural net to classify individual cell images.
    - Standardization - all the cells may have different intensities / contrast with the background.
    - Class imbalance - likely to be different numbers of cells for each environment.
 
 4) Compare with taking convolutional output and stacking into recurrent layers to take advantage of known time relationships.
