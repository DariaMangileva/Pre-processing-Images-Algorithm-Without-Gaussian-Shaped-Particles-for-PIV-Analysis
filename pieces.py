import skimage
from skimage import io 
from skimage.color import rgb2hsv, hsv2rgb
import matplotlib.pyplot as plt
import numpy as np  
from tqdm import tqdm
import PIL
from PIL import Image, ImageDraw
import math
from sklearn.metrics import mean_squared_error
import threading
from multiprocessing import Process
import time
import numba
from numba import cuda, roc, jit, njit, prange
import os
# for i in range(40):
# 	X = []
# 	Y = []
# 	mass = np.load('./mass/{i}.npy'.format(i = i))


cadr = os.listdir('./кадры0')
N = len(cadr)-2
#mse = mean_squared_error(A, B)
mask = io.imread('./mask.jpg')
#mass2 = io.imread('/home/daria/Документы/1.10.20/experiment/0.jpg')
z = np.zeros(shape = mask.shape)
z1 = np.zeros(shape = mask.shape)
X = []
Y = []
for i in range(0,720,16):
	for j in range(0,1280,16):
		if mask[i,j][0] > 230:
			X.append(j)
			Y.append(i)
mass2 = np.zeros(shape = (720,1280))
for i in range(len(X)):
	mass2[Y[i], X[i]] = 255
io.imsave('./0.jpg', mass2)
massiv = np.zeros(shape = (N,720,1280,3))
for g in range(N):
	file = io.imread('./кадры0/{i}.jpg'.format(i = g))
	massiv[g] = file
@jit(parallel = True)	
def f(): 
	for i in prange(len(Y)):
		point = np.zeros(shape = (N,2))
		x = X[i]
		y = Y[i]
		s = 0
		while s < N-1:
			mass = massiv[s]
			mass1 = massiv[s+1]
			z1 = np.zeros(shape = mass.shape)
			z = np.zeros(shape = (len(X), 2))
			A = mass[(y-8):(y+8),(x-8):(x+8)]
			MSE = []
			X0 = []
			Y0 = []
			for j in range(y-17,y+17):
				for k in range(x-17,x+17):
					B = mass1[(j-8):(j+8),(k-8):(k+8)]
					if sum(A.shape) == 35 and sum(B.shape) == 35:
						mse0 = np.square(np.subtract(A[0], B[0])).mean()
						mse1 = np.square(np.subtract(A[1], B[1])).mean()
						mse2 = np.square(np.subtract(A[2], B[2])).mean()
						mse = (mse0+mse1+mse2)/3
						MSE.append(mse)
						X0.append(k)
						Y0.append(j)
					else:
						MSE.append(0)
						Y0.append(0)
						X0.append(0)
			m = min(MSE)
			ind = MSE.index(m)
			if Y0[ind] != 0 or X0[ind] != 0:
				point[s] = [Y0[ind],X0[ind]]

		# if i > 0:
		# 	mass2 = io.imread('./img01/{n}.jpg'.format(n = s))
		# 	c = np.argwhere(mass2 > 230)
		# 	for t in range(len(c[:,0])):
		# 		z1[c[t][0],c[t][1]] = 255
		# 	mass2 = np.load('./mass/{i}.npy'.format(i = s))
		# 	z = mass2
		# 	z[i] = [x,y]
		# else:
		# 	z[i] = [x,y]
		#np.save('./mass/{i}.npy'.format(i = s), z)

		#io.imsave('./img01/{n}.jpg'.format(n = s),z1)
			s += 1
			x = X0[ind]
			y = Y0[ind]
		s += 0
		#s += 0
		np.save('./mass/mass{i}.npy'.format(i = i), point)
		print(i)
	
	# def func():
	# 	proc = Process(target=f, args=(i,))
	# 	proc.start()
	# 	procs.append(proc)
	# 	if i%20 == 0 and i != 0:
	# 		while procs[-1].is_alive():
	# 			time.sleep(1)
f()
	

# an_array = np.array(range(len(Y)))
# threadsperblock = 32
# blockspergrid = (an_array.size + (threadsperblock - 1)) // threadsperblock



	# proc = Process(target=func, args=(i,))
	# proc.start()
	# procs.append(proc)
	# if i%20 == 0 and i != 0:
	# 	minuts = []
	# 	while procs[-1].is_alive():
	# 		minuts.append(1)
	# 		time.sleep(1)
	# 	print(sum(minuts))


	# array = np.array([X00,Y00]).T
	# np.save('./mass/{i}.npy'.format(i = n), array)





	


# for i in tqdm(range(720)):
# 	for j in range(1280):
# 		if mask[i,j] == 255:
# 			c = np.argwhere((mass1[:,:,0] <= mass[i,j][0] + 2) & (mass1[:,:,0] >= mass[i,j][0] - 2) & (mass1[:,:,1] <= mass[i,j][1]+2) & (mass1[:,:,1] >= mass[i,j][1]-2) & (mass1[:,:,2] <= mass[i,j][2]+2) & (mass1[:,:,2] >= mass[i,j][2]-2))
# 			C = []
# 			for k in range(len(c[:,1])):
# 				if math.sqrt((i - c[:,0][k])**2 + (j - c[:,1][k])**2) < 5:
# 					C.append(c[:,0][k])
# 					if len(C) == 1: 
# 						a = c[:,0][k]
# 						b = c[:,1][k]
# 						if mass1[a+1,b][0] - 2 <= mass[i+1,j][0] and mass[i+1,j][0] <= mass1[a+1,b][0] + 2 and mass1[a+1,b][1] - 2 <= mass[i+1,j][1] and mass[i+1,j][1] <= mass1[a+1,b][1] + 2 and mass1[a+1,b][2] - 2 <= mass[i+1,j][1] and mass[i+1,j][2] <= mass1[a+1,b][2] + 2 and mass1[a+1,b+1][0] - 2 <= mass[i+1,j+1][0] and mass[i+1,j+1][0] <= mass1[a+1,b+1][0] + 2 and mass1[a+1,b+1][1] - 2 <= mass[i+1,j+1][1] and mass[i+1,j+1][1] <= mass1[a+1,b+1][1] + 2 and mass1[a+1,b+1][2] - 2 <= mass[i+1,j+1][2] and mass[i+1,j+1][2] <= mass1[a+1,b+1][2] + 2 and mass1[a,b+1][0] - 2 <= mass[i,j+1][0] and mass[i,j+1][0] <= mass1[a,b+1][0] + 2 and mass1[a,b+1][1] - 2 <= mass[i,j+1][1] and mass[i,j+1][1] <= mass1[a,b+1][1] + 2 and mass1[a,b+1][2] - 2 <= mass[i,j+1][2] and mass[i,j+1][2] <= mass1[a,b+1][2] + 2:
# 							z[i,j] = 255
# 							z1[a,b] = 255
# io.imsave('/home/daria/Документы/1.10.20/experiment/1/0.jpg',z)
# io.imsave('/home/daria/Документы/1.10.20/experiment/1/1.jpg',z1)

# im = Image.open('/home/daria/Документы/1.10.20/experiment/1.jpg')
# im.resize([150,150]).save('/home/daria/Документы/1.10.20/experiment/1.jpg')
# hsv_img = rgb2hsv(rgb_img)
# hsv_img[:,:,2] = hsv_img[:,:,2] + 0.9
# a = np.argwhere(hsv_img[:,:,2] < 0.6)
# for i in tqdm(a[:,0]):
# 	for j in a[:,1]:
# 		hsv_img[i,j,2] = 1
# rgb_img = hsv2rgb(hsv_img)
# plt.imshow(mask)
# plt.show()

