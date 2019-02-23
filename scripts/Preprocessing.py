import numpy as np
import math
from pywt import threshold, wavedec, Wavelet, waverec
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter

class Preprocessing():
	def mean_center(self, data):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		axis = 0 if data.ndim == 1 else 1
		mean = np.mean(data, axis=axis)

		if data.ndim == 1:
			new_data = data - mean

		else:
			mean = np.array([mean]*data.shape[0]).transpose()
			new_data = data - mean

		return new_data

	def moving_average(self, data, w_length):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		axis = 0 if data.ndim == 1 else 1 

		kernel 		= np.ones(w_length, dtype='uint8')/w_length
		new_data 	= convolve1d(data, kernel, axis=axis, mode='constant')
		return new_data

	def wavelet_denoising(self, data, wname, l):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		def get_default_thrs(axis):
			detail_coeffs 	= wavedec(data, wavelet='db1', level=1, axis=axis)[1]
			noise_level 	= np.median(abs(detail_coeffs), axis=axis)
			thrs 			= noise_level*math.sqrt(2*math.log(data.shape[axis]))/0.6745
			return thrs

		def decompose_data(wavelet):
			return wavedec(data, wavelet, level=l)

		def apply_thr(coeffs, thrs):
			coeffs_array 	= np.array(coeffs)
			app_coeffs 		= coeffs_array[0].copy()

			if data.shape[0] == 1:
				coeffs_array 		= threshold(coeffs_array, thrs, 'soft')
				coeffs_array[0] 	= app_coeffs

			else:
				for i in range(data.shape[0]):
					coeffs_array[:,i] 		= threshold(coeffs_array[:,i], thrs[i], 'soft')
					coeffs_array[:,i][0] 	= app_coeffs[i]

			return list(coeffs_array)

		def reconstruct_coeffs(thr_coeffs, wavelet):
			if (data.shape[0] == 1):
				return waverec(thr_coeffs, wavelet)[:-1]
			else:
				return waverec(thr_coeffs, wavelet)[:,:-1]

		axis 		= 0 if data.ndim == 1 else 1
		wavelet 	= Wavelet(wname)

		thrs 		= get_default_thrs(axis)
		coeffs 		= decompose_data(wavelet)
		thr_coeffs	= apply_thr(coeffs, thrs)
		thr_data 	= reconstruct_coeffs(thr_coeffs, wavelet)

		return thr_data

	def sav_gol(self, data, p_order, w_length):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		axis 		= 0 if data.ndim == 1 else 1

		half_size 	= int((w_length-1)/2)
		new_data 	= savgol_filter(data, window_length=w_length, polyorder=p_order, deriv=0, axis=axis)

		if data.ndim == 1:
			new_data[:half_size-1] 	= 0
			new_data[-half_size:] 	= 0

		else:
			new_data[:,:half_size-1] 	= 0
			new_data[:,-half_size:] 	= 0

		return new_data

	def sav_gol_derivative(self, data, d_order, p_order, w_length):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		axis 		= 0 if data.ndim == 1 else 1

		half_size 	= int((w_length-1)/2)
		new_data 	= savgol_filter(data, window_length=w_length, polyorder=p_order, deriv=d_order, axis=axis)

		if data.ndim == 1:
			new_data[:half_size-1] 	= 0
			new_data[-half_size:] 	= 0

		else:
			new_data[:,:half_size-1] 	= 0
			new_data[:,-half_size:] 	= 0

		return new_data

	def msc(self, data):
		assert data.ndim > 1, "Não é aceita uma amostra única."

		def fit_reg_line(row, mean):
			return np.polyfit(mean, row, 1)

		def apply_correction(coeffs):
			new_data = np.zeros((data.shape[0],data.shape[1]))

			for i in range(data.shape[0]):
				new_data[i,:] = (data[i,:]-coeffs[i,1])/coeffs[i,0]

			return new_data

		mean 		= np.mean(data, axis=0)
		coeffs 		= np.apply_along_axis(fit_reg_line, 1, data, mean)
		new_data 	= apply_correction(coeffs)
		
		return new_data

	def snv(self, data):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		def apply_correction(mean, std):
			new_data = (data.transpose()-mean)/std

			return new_data.transpose()

		axis 		= 0 if data.ndim == 1 else 1

		mean 		= np.mean(data, axis=axis)
		std 		= np.std(data, axis=axis)
		new_data	= apply_correction(mean, std)

		return new_data
