import numpy as np

class PreProcessing():

	def __init__(self, n_samples, n_variables):
		self.n_samples 		= n_samples
		self.n_variables 	= n_variables

	def mean_center(data):

	def moving_average(data, window_size):

		def convolve_func(row, kernel):
			return np.convolve(row, kernel, 'same')
		
		kernel 		= np.ones(window_size, dtype='uint8')/window_size
		new_data 	= np.apply_along_axis(convolve_func, 1, data, kernel)
		return new_data

	def wavelet(data, wname, level, thr):

	def sav_gol(data, d_order, p_order, window_size):

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
