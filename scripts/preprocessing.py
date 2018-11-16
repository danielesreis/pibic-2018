import numpy as np
from pywt import threshold, wavedec
from scipy.ndimage import convolve1d
from scipy.signal import savgol_filter

class PreProcessing():

	def __init__(self, n_samples, n_variables):
		self.n_samples 		= n_samples
		self.n_variables 	= n_variables

	def mean_center(self, data):
		mean_values = np.mean(data, axis=0)
		new_data 	= data - mean_values
		return new_data

	def moving_average(self, data, w_length):
		kernel 		= np.ones(w_length, dtype='uint8')/w_length
		new_data 	= convolve1d(data, kernel, axis=1, mode='constant')
		return new_data

	def wavelet_denoising(self, data, wname, l):
		# assert len(data.shape) > 2, "Matrizes com mais de 2 dimensões não são aceitas"

		def get_default_thrs(self, data, axis):
			detail_coeffs 	= wavedec(data, wavelet='db1', level=1, axis=axis)[1]
			noise_level 	= np.median(abs(detail_coeffs), axis=axis)
			thrs 			= noise_level*math.sqrt(2*math.log(data.shape[axis]))/0.6745
			return thrs

		def apply_thr(oeffs, thr):
			# return pywt.threshold(coeffs, thr, 'soft')

		axis 		= 0 if len(data.shape) == 1 else 1
		thrs 		= get_default_thrs(data, axis)
		new_data	= apply_thr(coeffs, thr)
		return new_data

	def sav_gol(self, data, d_order, p_order, w_length):
		half_size 	= int((w_length-1)/2)

		new_data 	= savgol_filter(data, window_length=w_length, polyorder=p_order, deriv=d_order, axis=1)

		new_data[:, :half_size-1] 	= 0
		new_data[:, -half_size:] 	= 0

		return new_data

	def msc(self, data):

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

	def snv(self, data):

		def apply_correction(mean, std):
			new_data = np.zeros((self.n_samples,self.n_variables))

			for i in range(self.n_samples):
				new_data[i,:] = (data[i,:]-mean[i])/std[i]

			return new_data

		mean 		= np.mean(data, axis=1)
		std 		= np.std(data, axis=1)
		new_data	= apply_correction(mean, std)

		return new_data
