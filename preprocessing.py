import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter

class PreProcessing():

	def __init__(self, n_samples, n_variables):
		self.n_samples 		= n_samples
		self.n_variables 	= n_variables

	def mean_center(data):
		mean_values =  np.mean(data, axis=0)
		new_data 	= data - mean_values
		return new_data

	def moving_average(data, w_length):
		kernel 		= np.ones(w_length, dtype='uint8')/w_length
		new_data 	= convolve1d(data, kernel, axis=1, mode='constant')
		return new_data

	def wavelet_denoising(data, wname, l):

		def get_default_thr():

		def apply_thr(coeffs, thr):

		coeffs 		= wavedec(data, wavelet=wname, level=l, axis=1) 
		thr 		= get_default_thr():
		new_data	= apply_thr(coeffs, thr)

	def sav_gol(data, d_order, p_order, w_length):
		half_size 	= int((w_length-1)/2)

		new_data 	= savgol_filter(data, window_length=w_length, polyorder=p_order, deriv=d_order, axis=1)

		new_data[:, :half_size-1] 	= 0
		new_data[:, -half_size:] 	= 0

		return new_data

	def msc(data):

		def fit_reg_line(row, mean):
			return np.polyfit(mean, row, 1)

		def apply_correction(coeffs):
			new_data = np.zeros((self.n_samples,self.n_variables))

			for i in range(self.n_samples):
				new_data[i,:] = (data[i,:]-coeffs[i,1])/coeffs[i,0]

			return new_data

		mean 		= np.mean(data, axis=0)
		coeffs 		= np.apply_along_axis(fit_reg_line, 1, data, mean)
		new_data 	= apply_correction(coeffs)
		
		return new_data

	def snv(data):

		def apply_correction(mean, std):
			new_data = np.zeros((self.n_samples,self.n_variables))

			for i in range(self.n_samples):
				new_data[i,:] = (data[i,:]-mean[i])/std[i]

			return new_data

		mean 		= np.mean(data, axis=1)
		std 		= np.std(data, axis=1)
		new_data	= apply_correction(mean, std)

		return new_data
