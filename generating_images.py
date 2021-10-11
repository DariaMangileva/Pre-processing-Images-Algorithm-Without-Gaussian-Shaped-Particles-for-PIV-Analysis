import numpy as np 
import os
import skimage
from skimage import io 
#mass = np.load('./mass/0.npy')
point = os.listdir('./mass')
cadr = os.listdir('./кадры0')
for i in range(len(cadr)):
	z = np.zeros(shape = (720,1280))
	for j in range(len(point)):
		try:
			cords = np.load('./mass/{i}'.format(i = point[j]))
			x,y = cords[i]
			z[int(x),int(y)] = 1
		except ValueError:
			continue
	io.imsave('./point/{i}.jpg'.format(i = i),z)

