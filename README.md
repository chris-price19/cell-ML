# cell-ML
Image classification of cell environments ENM 531


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
