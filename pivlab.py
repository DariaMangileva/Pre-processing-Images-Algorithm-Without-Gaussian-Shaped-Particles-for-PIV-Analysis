
from openpiv import tools, pyprocess, validation, filters, scaling
import numpy as np
import matplotlib.pyplot as plt
import imageio
import PIL
from PIL import Image 
import shutil
	# im = io.imread('{path}/framje{i}.jpg'.format(path = path, file = i))
import skimage
from skimage import io
import os
winsize = 16 # pixels, interrogation window size in frame A
searchsize = 32 # pixels, search in image B
overlap = 8 # pixels, 50% overlap
dt = 0.02 # sec, time interval between pulses
import skimage as io 
from skimage.color import rgb2gray
from tqdm import tqdm
# path1 = '/home/daria/Документы/1.10.20/experiment/image'
# path2 = '/home/daria/Документы/1.10.20/experiment/mask'
# mass1 = os.listdir('/home/daria/Документы/1.10.20/experiment/image')
# mass2 = os.listdir('/home/daria/Документы/1.10.20/experiment/mask')
#mass3 = os.listdir('/home/daria/Документы/1.10.20/experiment/video')
def func(dir_, dir_txt):
	m = os.listdir('{n}'.format(n = dir_))
#os.mkdir('./image1')
	for i in tqdm(range(len(m)-2)):
	# im = Image.open('/home/daria/Документы/1.10.20/experiment/mask/{i}'.format(i = mass2[i]))
	# im.resize([48,27]).save('/home/daria/Документы/1.10.20/experiment/mask/{i}'.format(i = mass2[i]))
	#os.rename('/home/daria/Документы/1.10.20/experiment/image/{file}'.format(file = mass1[i]), '/home/daria/Документы/1.10.20/experiment/image/{file}.jpg'.format(file = i))
	#shutil.move('/home/daria/Документы/1.10.20/experiment/mask/{file}'.format(file = mass1[i]), '/home/daria/Документы/1.10.20/experiment/mask2/{file}'.format(file = mass1[i]))

		frame_a  = tools.imread( '{n}/{i}.jpg'.format(i = i, n = dir_))
		frame_b  = tools.imread( '{n}/{i}.jpg'.format(i = i + 1, n = dir_))
		filename = '{n}/{i}.txt'.format(i = i, n = dir_txt)
		u0, v0, sig2noise = pyprocess.extended_search_area_piv(frame_a.astype(np.int32), 
                                                       		frame_b.astype(np.int32), 
                                                       		window_size=winsize, 
                                                       		overlap=overlap, 
                                                       		dt=dt, 
                                                       		search_area_size=searchsize, 
                                                       		sig2noise_method='peak2peak')
		x, y = pyprocess.get_coordinates( image_size=frame_a.shape, 
                                 	window_size=searchsize, 
                                   	overlap=overlap )
		u1, v1, mask = validation.sig2noise_val( u0, v0, 
                                        	sig2noise, 
                                        	threshold = 1.05 )
	# mask = io.imread('/home/daria/Документы/1.10.20/experiment/mask2/results{i}.jpg'.format(i = i))
	# m = np.zeros(shape = [29,53], dtype = bool)
	# for i in range(28):
	# 	for j in range(52):
	# 		if mask.shape[-1] == 3:
	# 			if mask[i,j][0] > 200:
	# 				m[i,j] = True
	# 		else:
	# 			if mask[i,j] > 200:
	# 				m[i,j] = True
	# mask = m
		u2, v2 = filters.replace_outliers( u1, v1, 
                                  	method='localmean', 
                                  	max_iter=3, 
                                  	kernel_size=3)
		x, y, u3, v3 = scaling.uniform(x, y, u2, v2, 
                                	scaling_factor = 50) # 96.52 microns/pixel
		tools.save(x, y, u3, v3, mask = mask, filename = filename)
#func(dir_ = './canny', dir_txt = './txt_canny')
#func(dir_ = './point', dir_txt = './txt')
#func(dir_ = './sobel', dir_txt = './txt_sobel')
func(dir_ = './кадры0', dir_txt = './txt_raw')
	# fig, ax = plt.subplots(figsize=(8,8))
	# tools.display_vector_field('./txt_sobel/{k}.txt'.format(k = i), 
 #                           		ax=ax, scaling_factor=100, 
 #                           		scale=50, # scale defines here the arrow length
 #                           		width=0.0035, # width is the thickness of the arrow
 #                           		on_img=True, # overlay on the image
 #                           		image_name='./mask.jpg')
	#fig.savefig('./image/{n}.jpg'.format(n = i))
	#mass = io.imread('/home/daria/Документы/1.10.20/experiment/image/0.jpg')
	
