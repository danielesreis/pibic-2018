
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import math


class Training():

	def __init__(self, max_iter, scale=True):
		self.max_iter = max_iter
		self.scale = scale
		self._lv_number = None

	@property
	def lv_number(self):
		return self._lv_number

	@lv_number.setter
	def lv_number(self, lv_number):
		self._lv_number = lv_number

	def optimal_lv_num(self, X, Y):
		rmsecv = np.array([])
		for lv in np.arange(10):
			pls = PLSRegression(lv, self.max_iter, self.scale)
			rmsecv = np.append(rmsecv, math.sqrt(abs(cross_val_score(pls, X, Y, scoring='neg_mean_squared_error', cv=10).mean())))

		optimal = np.where(rmsecv==min(rmsecv))[0][0] + 1
		self.lv_number = optimal

	def pls(self):
		return PLSRegression(self.lv_number, self.max_iter, self.scale)

	def pls_cv(self, X, Y, k):
		pls = self.pls()
		kf = KFold(n_splits=k, shuffle=True, random_state=42)
		metrics = pd.DataFrame(columns=['R2', 'MSE'])

		for train_i, test_i in kf.split(X):
			X_train, X_test = X.iloc[train_i], X.iloc[test_i]
			Y_train, Y_test = Y.iloc[train_i], Y.iloc[test_i]
			pls.fit(X_train, Y_train)
			Y_pred = pls.predict(X_test)
			r2 = r2_score(Y_test, Y_pred)
			mse = mean_squared_error(Y_test, Y_pred)
			metrics = metrics.append({'R2': r2, 'MSE': mse}, ignore_index=True)

		metrics = metrics.append(metrics.mean(), ignore_index=True)
		indexes = list(metrics.index[:-1].values+1) + ['mean']
		metrics.index = indexes
		return metrics

	def build(self):

	def predict(self, model, X_test, Y_test):
		Y_pred = model.predict(X_test)

		r2 = r2_score(Y_test, Y_pred)
		mse = mean_squared_error(Y_test, Y_pred)

		return r2, mse
