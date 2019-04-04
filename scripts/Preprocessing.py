import numpy as np
import math
from pywt import threshold, wavedec, Wavelet, waverec
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter

class Preprocessing():

	def mean_center(self, data):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		def subtract_mean(data, mean):
			return data - mean

		axis = 0 if data.ndim == 1 else 1
		mean = np.mean(data, axis=axis)

		if data.ndim == 1:
			new_data = data - mean

		else:
			mean = mean.transpose()
			new_data = np.apply_along_axis(subtract_mean, 0, data, mean)

		return new_data

	def moving_average(self, data, w_length):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		axis = 0 if data.ndim == 1 else 1 

		kernel 		= np.ones(w_length, dtype='uint8')/w_length
		new_data 	= convolve1d(data, kernel, axis=axis, mode='constant')
		return new_data

	def wavelet_denoising(self, data, wname, l):
		assert data.ndim <= 2, "Matrizes com mais de 2 dimensões não são aceitas."

		def get_default_thrs(sample):
			detail_coeffs 	= wavedec(sample, wavelet='db1', level=1, axis=0)[1]
			noise_level 	= np.median(abs(detail_coeffs), axis=0)
			thrs 			= noise_level*math.sqrt(2*math.log(sample.shape[0]))/0.6745
			return thrs

		def decompose_data(sample, wavelet):
			return wavedec(sample, wavelet, level=l)

		def apply_thr(coeffs, thrs):
			app_coeffs 	= coeffs[0].copy()

			coeffs 		= list(map(lambda arr: threshold(arr, thrs, 'soft'), coeffs))
			coeffs[0] 	= app_coeffs

			return coeffs

		def reconstruct_coeffs(thr_coeffs, wavelet):
			return waverec(thr_coeffs, wavelet)	

		def denoise(sample, wavelet):
			thrs 		= get_default_thrs(sample)
			coeffs 		= decompose_data(sample, wavelet)
			thr_coeffs	= apply_thr(coeffs, thrs)
			thr_data 	= reconstruct_coeffs(thr_coeffs, wavelet)
			return thr_data
			
		wavelet 	= Wavelet(wname)

		if data.ndim == 1:
			thr_data = denoise(data, wavelet)
		else:
			thr_data = np.apply_along_axis(denoise, 1, data, wavelet)

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
