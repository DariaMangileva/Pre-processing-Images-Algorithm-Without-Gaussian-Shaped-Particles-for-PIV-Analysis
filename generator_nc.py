"""
Generate a netCDF file with a vortex field.
"""
import numpy as np 
import skimage
from skimage import io 
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from tqdm import tqdm
import fitting
import os 
filename = './txt_canny'
nn = os.listdir(filename)
for n in tqdm(range(len(nn))):
	args = []
	mass = np.loadtxt('./txt_canny/{n}.txt'.format(n = n))
	string = np.where(mass[:,-1] == 1)
	string0 = np.where(mass[:,-1] == 0)
	mask = io.imread('./mask.jpg')
	X = []
	Y = []
	U = []
	V = []
	for i in range(len(mass[:,-1])):
		if mass[i][-1] == 1:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(mass[i][2])
				Y.append(int(mass[i][1]*50))
				V.append(mass[i][3])
		else:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(0)
				Y.append(int(mass[i][1]*50))
				V.append(0)
	for i in range(len(X)):
		if np.isnan(U[i]) == True:
			U[i] = 0
			V[i] = 0
	Y0 = list(set(Y))
	X0 = list(set(X))
	N0 = []
	I = []
	for i in Y0:
		N = []
		for j in Y:
			if j == i:
				N.append(1)
		I.append(i)
		N0.append(sum(N))
	N0x = []
	Ix = []
	for i in X0:
		N = []
		for j in X:
			if j == i:
				N.append(1)
		Ix.append(i)
		N0x.append(sum(N))
	ind_y = max(N0)
	ind_x = max(N0x)
	u = np.zeros(shape = (35,35))
	X00 = list(np.sort(X0))
	Y00 = list(np.sort(Y0))
	X_new = [X00.index(i) for i in X]
	Y_new = [Y00.index(i) for i in Y]
	for i in range(len(Y_new)):
		u[Y_new[i],X_new[i]] = U[i]
	v = np.zeros(shape = (35,35))
	for i in range(len(Y_new)):
		v[Y_new[i],X_new[i]] = V[i]



		







	if __name__ == '__main__':

		parser = argparse.ArgumentParser(description='generate a vortex field in a netCDF file',
                                     formatter_class=argparse.RawTextHelpFormatter)

		parser.add_argument('-o', '--output', dest='outfile', type=str,
                        help='output NetCDF file', metavar='FILE',
                        default='./nc_canny/{n}.nc'.format(n = n))

		parser.add_argument('-ndim', '--ndim', dest='ndim', type=int,
                        help='spatial mesh dimension, for each x and y variables',
                        default=256)

		args = parser.parse_args()

	print('Generating {:s} file with a {:d}x{:d} mesh'.format(args.outfile, args.ndim, args.ndim))

# Try to write the file
	try:

		datafile_write = netCDF4.Dataset(args.outfile, 'w', format='NETCDF4')
	except IOError:
		print('There was an error writing the file!')
		sys.exit()

	datafile_write.description = 'Sample field with an Oseen vortex'

	ndim = 35
	  # spacing
# dimensions
	datafile_write.createDimension('resolution_x', ndim)
	datafile_write.createDimension('resolution_y', ndim)
	datafile_write.createDimension('resolution_z', 1)

# variables
	velocity_x = datafile_write.createVariable('velocity_x', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_y = datafile_write.createVariable('velocity_y', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_z = datafile_write.createVariable('velocity_z', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))

#data
	velocity_x[:] = np.array([u])
	velocity_y[:] = np.array([v])
	velocity_z[:] = np.random.random((1, ndim, ndim))/10

	x_grid = np.linspace(0, ndim, ndim)
	y_grid = np.linspace(0, ndim, ndim)

	x_matrix, y_matrix = np.meshgrid(x_grid, y_grid)
	# core_radius = 5.0
	# gamma = 30
	# x_center = 64
	# y_center = 192
	# u_advection = 0.0
	# v_advection = 0.3
	# u_data, v_data = fitting.velocity_model(core_radius, gamma, x_center, y_center, u_advection, v_advection, x_matrix,
 #                                        y_matrix)
	#u_data = u_data + u_advection
	#v_data = v_data + v_advection

	# velocity_x[0, :, :] += u_data[:, :]
	# velocity_y[0, :, :] += v_data[:, :]
	s = 1  # sampling factor for quiver plot
	plt.quiver(x_matrix[::s, ::s], y_matrix[::s, ::s], velocity_x[0, ::s, ::s], velocity_y[0, ::s, ::s])

	plt.savefig('./vector_canny/{n}.png'.format(n = n))
	plt.close()


filename = './txt'
nn = os.listdir(filename)
for n in tqdm(range(len(nn))):
	args = []
	mass = np.loadtxt('./txt/{n}.txt'.format(n = n))
	string = np.where(mass[:,-1] == 1)
	string0 = np.where(mass[:,-1] == 0)
	mask = io.imread('./mask.jpg')
	X = []
	Y = []
	U = []
	V = []
	for i in range(len(mass[:,-1])):
		if mass[i][-1] == 1:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(mass[i][2])
				Y.append(int(mass[i][1]*50))
				V.append(mass[i][3])
		else:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(0)
				Y.append(int(mass[i][1]*50))
				V.append(0)
	for i in range(len(X)):
		if np.isnan(U[i]) == True:
			U[i] = 0
			V[i] = 0
	Y0 = list(set(Y))
	X0 = list(set(X))
	N0 = []
	I = []
	for i in Y0:
		N = []
		for j in Y:
			if j == i:
				N.append(1)
		I.append(i)
		N0.append(sum(N))
	N0x = []
	Ix = []
	for i in X0:
		N = []
		for j in X:
			if j == i:
				N.append(1)
		Ix.append(i)
		N0x.append(sum(N))
	ind_y = max(N0)
	ind_x = max(N0x)
	u = np.zeros(shape = (35,35))
	X00 = list(np.sort(X0))
	Y00 = list(np.sort(Y0))
	X_new = [X00.index(i) for i in X]
	Y_new = [Y00.index(i) for i in Y]
	for i in range(len(Y_new)):
		u[Y_new[i],X_new[i]] = U[i]
	v = np.zeros(shape = (35,35))
	for i in range(len(Y_new)):
		v[Y_new[i],X_new[i]] = V[i]



		







	if __name__ == '__main__':

		parser = argparse.ArgumentParser(description='generate a vortex field in a netCDF file',
                                     formatter_class=argparse.RawTextHelpFormatter)

		parser.add_argument('-o', '--output', dest='outfile', type=str,
                        help='output NetCDF file', metavar='FILE',
                        default='./nc/{n}.nc'.format(n = n))

		parser.add_argument('-ndim', '--ndim', dest='ndim', type=int,
                        help='spatial mesh dimension, for each x and y variables',
                        default=256)

		args = parser.parse_args()

	print('Generating {:s} file with a {:d}x{:d} mesh'.format(args.outfile, args.ndim, args.ndim))

# Try to write the file
	try:

		datafile_write = netCDF4.Dataset(args.outfile, 'w', format='NETCDF4')
	except IOError:
		print('There was an error writing the file!')
		sys.exit()

	datafile_write.description = 'Sample field with an Oseen vortex'

	ndim = 35
	  # spacing
# dimensions
	datafile_write.createDimension('resolution_x', ndim)
	datafile_write.createDimension('resolution_y', ndim)
	datafile_write.createDimension('resolution_z', 1)

# variables
	velocity_x = datafile_write.createVariable('velocity_x', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_y = datafile_write.createVariable('velocity_y', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_z = datafile_write.createVariable('velocity_z', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))

#data
	velocity_x[:] = np.array([u])
	velocity_y[:] = np.array([v])
	velocity_z[:] = np.random.random((1, ndim, ndim))/10

	x_grid = np.linspace(0, ndim, ndim)
	y_grid = np.linspace(0, ndim, ndim)

	x_matrix, y_matrix = np.meshgrid(x_grid, y_grid)
	# core_radius = 5.0
	# gamma = 30
	# x_center = 64
	# y_center = 192
	# u_advection = 0.0
	# v_advection = 0.3
	# u_data, v_data = fitting.velocity_model(core_radius, gamma, x_center, y_center, u_advection, v_advection, x_matrix,
 #                                        y_matrix)
	#u_data = u_data + u_advection
	#v_data = v_data + v_advection

	# velocity_x[0, :, :] += u_data[:, :]
	# velocity_y[0, :, :] += v_data[:, :]
	s = 1  # sampling factor for quiver plot
	plt.quiver(x_matrix[::s, ::s], y_matrix[::s, ::s], velocity_x[0, ::s, ::s], velocity_y[0, ::s, ::s])

	plt.savefig('./vector/{n}.png'.format(n = n))
	plt.close()


filename = './txt_sobel'
nn = os.listdir(filename)
for n in tqdm(range(len(nn))):
	args = []
	mass = np.loadtxt('./txt_sobel/{n}.txt'.format(n = n))
	string = np.where(mass[:,-1] == 1)
	string0 = np.where(mass[:,-1] == 0)
	mask = io.imread('./mask.jpg')
	X = []
	Y = []
	U = []
	V = []
	for i in range(len(mass[:,-1])):
		if mass[i][-1] == 1:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(mass[i][2])
				Y.append(int(mass[i][1]*50))
				V.append(mass[i][3])
		else:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(0)
				Y.append(int(mass[i][1]*50))
				V.append(0)
	for i in range(len(X)):
		if np.isnan(U[i]) == True:
			U[i] = 0
			V[i] = 0
	Y0 = list(set(Y))
	X0 = list(set(X))
	N0 = []
	I = []
	for i in Y0:
		N = []
		for j in Y:
			if j == i:
				N.append(1)
		I.append(i)
		N0.append(sum(N))
	N0x = []
	Ix = []
	for i in X0:
		N = []
		for j in X:
			if j == i:
				N.append(1)
		Ix.append(i)
		N0x.append(sum(N))
	ind_y = max(N0)
	ind_x = max(N0x)
	u = np.zeros(shape = (35,35))
	X00 = list(np.sort(X0))
	Y00 = list(np.sort(Y0))
	X_new = [X00.index(i) for i in X]
	Y_new = [Y00.index(i) for i in Y]
	for i in range(len(Y_new)):
		u[Y_new[i],X_new[i]] = U[i]
	v = np.zeros(shape = (35,35))
	for i in range(len(Y_new)):
		v[Y_new[i],X_new[i]] = V[i]



		







	if __name__ == '__main__':

		parser = argparse.ArgumentParser(description='generate a vortex field in a netCDF file',
                                     formatter_class=argparse.RawTextHelpFormatter)

		parser.add_argument('-o', '--output', dest='outfile', type=str,
                        help='output NetCDF file', metavar='FILE',
                        default='./nc_sobel/{n}.nc'.format(n = n))

		parser.add_argument('-ndim', '--ndim', dest='ndim', type=int,
                        help='spatial mesh dimension, for each x and y variables',
                        default=256)

		args = parser.parse_args()

	print('Generating {:s} file with a {:d}x{:d} mesh'.format(args.outfile, args.ndim, args.ndim))

# Try to write the file
	try:

		datafile_write = netCDF4.Dataset(args.outfile, 'w', format='NETCDF4')
	except IOError:
		print('There was an error writing the file!')
		sys.exit()

	datafile_write.description = 'Sample field with an Oseen vortex'

	ndim = 35
	  # spacing
# dimensions
	datafile_write.createDimension('resolution_x', ndim)
	datafile_write.createDimension('resolution_y', ndim)
	datafile_write.createDimension('resolution_z', 1)

# variables
	velocity_x = datafile_write.createVariable('velocity_x', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_y = datafile_write.createVariable('velocity_y', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_z = datafile_write.createVariable('velocity_z', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))

#data
	velocity_x[:] = np.array([u])
	velocity_y[:] = np.array([v])
	velocity_z[:] = np.random.random((1, ndim, ndim))/10

	x_grid = np.linspace(0, ndim, ndim)
	y_grid = np.linspace(0, ndim, ndim)

	x_matrix, y_matrix = np.meshgrid(x_grid, y_grid)
	# core_radius = 5.0
	# gamma = 30
	# x_center = 64
	# y_center = 192
	# u_advection = 0.0
	# v_advection = 0.3
	# u_data, v_data = fitting.velocity_model(core_radius, gamma, x_center, y_center, u_advection, v_advection, x_matrix,
 #                                        y_matrix)
	#u_data = u_data + u_advection
	#v_data = v_data + v_advection

	# velocity_x[0, :, :] += u_data[:, :]
	# velocity_y[0, :, :] += v_data[:, :]
	s = 1  # sampling factor for quiver plot
	plt.quiver(x_matrix[::s, ::s], y_matrix[::s, ::s], velocity_x[0, ::s, ::s], velocity_y[0, ::s, ::s])

	plt.savefig('./vector_sobel/{n}.png'.format(n = n))
	plt.close()

filename = './txt_raw'
nn = os.listdir(filename)
for n in tqdm(range(len(nn))):
	args = []
	mass = np.loadtxt('./txt_raw/{n}.txt'.format(n = n))
	string = np.where(mass[:,-1] == 1)
	string0 = np.where(mass[:,-1] == 0)
	mask = io.imread('./mask.jpg')
	X = []
	Y = []
	U = []
	V = []
	for i in range(len(mass[:,-1])):
		if mass[i][-1] == 1:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(mass[i][2])
				Y.append(int(mass[i][1]*50))
				V.append(mass[i][3])
		else:
			if mask[int(mass[i][1]*50),int(mass[i][0]*50)][0] > 230:
				X.append(int(mass[i][0]*50))
				U.append(0)
				Y.append(int(mass[i][1]*50))
				V.append(0)
	for i in range(len(X)):
		if np.isnan(U[i]) == True:
			U[i] = 0
			V[i] = 0
	Y0 = list(set(Y))
	X0 = list(set(X))
	N0 = []
	I = []
	for i in Y0:
		N = []
		for j in Y:
			if j == i:
				N.append(1)
		I.append(i)
		N0.append(sum(N))
	N0x = []
	Ix = []
	for i in X0:
		N = []
		for j in X:
			if j == i:
				N.append(1)
		Ix.append(i)
		N0x.append(sum(N))
	ind_y = max(N0)
	ind_x = max(N0x)
	u = np.zeros(shape = (35,35))
	X00 = list(np.sort(X0))
	Y00 = list(np.sort(Y0))
	X_new = [X00.index(i) for i in X]
	Y_new = [Y00.index(i) for i in Y]
	for i in range(len(Y_new)):
		u[Y_new[i],X_new[i]] = U[i]
	v = np.zeros(shape = (35,35))
	for i in range(len(Y_new)):
		v[Y_new[i],X_new[i]] = V[i]



		







	if __name__ == '__main__':

		parser = argparse.ArgumentParser(description='generate a vortex field in a netCDF file',
                                     formatter_class=argparse.RawTextHelpFormatter)

		parser.add_argument('-o', '--output', dest='outfile', type=str,
                        help='output NetCDF file', metavar='FILE',
                        default='./nc_raw/{n}.nc'.format(n = n))

		parser.add_argument('-ndim', '--ndim', dest='ndim', type=int,
                        help='spatial mesh dimension, for each x and y variables',
                        default=256)

		args = parser.parse_args()

	print('Generating {:s} file with a {:d}x{:d} mesh'.format(args.outfile, args.ndim, args.ndim))

# Try to write the file
	try:

		datafile_write = netCDF4.Dataset(args.outfile, 'w', format='NETCDF4')
	except IOError:
		print('There was an error writing the file!')
		sys.exit()

	datafile_write.description = 'Sample field with an Oseen vortex'

	ndim = 35
	  # spacing
# dimensions
	datafile_write.createDimension('resolution_x', ndim)
	datafile_write.createDimension('resolution_y', ndim)
	datafile_write.createDimension('resolution_z', 1)

# variables
	velocity_x = datafile_write.createVariable('velocity_x', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_y = datafile_write.createVariable('velocity_y', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))
	velocity_z = datafile_write.createVariable('velocity_z', 'f4', ('resolution_z', 'resolution_y', 'resolution_x'))

#data
	velocity_x[:] = np.array([u])
	velocity_y[:] = np.array([v])
	velocity_z[:] = np.random.random((1, ndim, ndim))/10

	x_grid = np.linspace(0, ndim, ndim)
	y_grid = np.linspace(0, ndim, ndim)

	x_matrix, y_matrix = np.meshgrid(x_grid, y_grid)
	# core_radius = 5.0
	# gamma = 30
	# x_center = 64
	# y_center = 192
	# u_advection = 0.0
	# v_advection = 0.3
	# u_data, v_data = fitting.velocity_model(core_radius, gamma, x_center, y_center, u_advection, v_advection, x_matrix,
 #                                        y_matrix)
	#u_data = u_data + u_advection
	#v_data = v_data + v_advection

	# velocity_x[0, :, :] += u_data[:, :]
	# velocity_y[0, :, :] += v_data[:, :]
	s = 1  # sampling factor for quiver plot
	plt.quiver(x_matrix[::s, ::s], y_matrix[::s, ::s], velocity_x[0, ::s, ::s], velocity_y[0, ::s, ::s])

	plt.savefig('./vector_raw/{n}.png'.format(n = n))
	plt.close()




