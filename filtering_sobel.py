import cv2
import skimage 
from skimage import io 
from skimage.filters import sobel, median, gaussian, threshold_otsu, try_all_threshold, laplace, roberts
from skimage.filters.rank import otsu
import matplotlib.pyplot as plt
import os 
from tqdm import tqdm
mass = os.listdir('./кадры0')
for i in tqdm(range(len(mass))):
	img = cv2.imread('./кадры0/{i}.jpg'.format(i = i),0)
	edges = sobel(img)
	edges = edges > 0.05
	io.imsave('./sobel/{i}.jpg'.format(i = i),edges)


