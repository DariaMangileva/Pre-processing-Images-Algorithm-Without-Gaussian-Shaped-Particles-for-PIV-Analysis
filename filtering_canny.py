import skimage 
from skimage import io 
import os
from skimage import feature
from skimage.color import rgb2gray
import matplotlib.pyplot as plt 
mass = os.listdir('./кадры0')
for i in range(len(mass)):
	im = io.imread('./кадры0/{n}'.format(n = mass[i]))
	im = rgb2gray(im)
	im = feature.canny(im, sigma=1)
	io.imsave('./canny/{n}'.format(n = mass[i]), im)