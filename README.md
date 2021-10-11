# Pre-processing-Images-Algorithm-Without-Gaussian-Shaped-Particles-for-PIV-Analysis

This repository was created for reproducibility the results of the research by Mangileva et al[1]. For correctly work of PIV analis it recomends to use module openpiv that uploaded in this repository as zip file. 

Moreover, for faster computing it recomends to use the more powerful GPU like Tesla v100.

The video files were uploaded on google drive: https://drive.google.com/drive/folders/1xTJ7y7QqNlv1y5ICWnLS_FSFknxGt-CA?usp=sharing. The fibrilation had been starting since MVI_0796.MOV.

It is important to note, the project is under development.

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

Description and right order of files for calculating SSC index[2]:

generator_nc.py - generating nc files

calculating_swirling.py - calculate the max value of SSC index for each vector field

 References
 1. 
 2. J. Zhou, R.J. Adrian, S. Balachandar, T.M. Kendall, Mechanisms for generating coherent packets
of hairpin vortices in channel flow, Journal of Fluid Mechanics 387 (1999): 353–96. doi:
http://dx.doi.org/10.1017/s002211209900467x







