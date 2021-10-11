import skimage 
from skimage import io 
import os 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def f(direct):
	mass = os.listdir('{n}'.format(n = direct))
	mask = io.imread('./mask.jpg')
	X = []
	Y = []
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i,j][0] == 0:
				X.append(j)
				Y.append(i)
	for i in tqdm(range(len(mass))):
		im = io.imread('{z}/{n}'.format(n = mass[i], z = direct))
		for j in range(len(X)):
			if len(im.shape) == 3:
				im[Y[j],X[j]] = [0,0,0]
			else:
				im[Y[j],X[j]] = 0
		io.imsave('{z}/{n}'.format(n = mass[i], z = direct), im)
f(direct = './canny')
f(direct = './sobel')
f(direct = './кадры0')
