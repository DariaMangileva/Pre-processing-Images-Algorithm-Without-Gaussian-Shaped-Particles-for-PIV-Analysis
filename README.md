# Pre-processing-Images-Algorithm-Without-Gaussian-Shaped-Particles-for-PIV-Analysis

This repository was created for reproducibility the results of the research by Mangileva et al[]

Description and right order of files for proposed method of preprocessing:

video_fragmentation.py - splitting video into frames

pieces.py - creating massives with coordinates of each points

generating_images.py - gemerating the images with this points

pivlab.py - PIV analise

Description and right order of files for classical filters of preprocessing:

video_fragmentation.py - splitting video into frames

filtering_canny or filtering_sobel - filterring each frame with filter

mask.py - making the area outside the heart black

pivlab.py - PIV analise

Description and right order of files for calculating SSC index:

generator_nc.py - generating nc files

calculating_swirling.py - calculate the max value of SSC index for each vector field




