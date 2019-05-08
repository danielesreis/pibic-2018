
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import math


class Training():

	def __init__(self, max_iter_pls, n_components, k, scale=True):
		self.max_iter = max_iter_pls
		self.n_components = n_components
		self.random_state = 0
		self.k = k
		self.scale = scale

		self.lv_number = None
		self.pls = None

		# print('These are the default RNA parameters... you need to optimize them!')
		self.hidden_layer_sizes = 5
		self.activation = 'logistic'
		self.learning_rate_init = 0.2
		self.max_iter = 100
		self.rna = None

	@property
	def lv_number(self):
		return self._lv_number

	@lv_number.setter
	def lv_number(self, lv_number):
		self._lv_number = lv_number

	@property
	def pls(self):
		return self._pls

	@pls.setter
	def pls(self, pls):
		self._pls = pls

	@property
	def rna(self):
		return self._rna

	@rna.setter
	def rna(self, rna):
		self._rna = rna

	def optimal_lv_num(self, X, Y):
		self.lv_number = 10
		return
		rmsecv = np.array([])
		for lv in np.arange(1, self.n_components+1):
			pls = PLSRegression(lv, self.max_iter, self.scale)
			rmsecv = np.append(rmsecv, math.sqrt(abs(cross_val_score(pls, X, Y, scoring='neg_mean_squared_error', cv=self.k).mean())))

		self.lv_number = np.where(rmsecv==min(rmsecv))[0][0] + 1

	def pls_model(self):
		self.pls = PLSRegression(self.lv_number, self.max_iter, self.scale)

	def rna_model(self):
		self.rna = MLPRegressor(tuple([self.hidden_layer_sizes]), self.activation, self.learning_rate_init, self.max_iter)

	def cv(self, model, X, Y):
		metrics = pd.DataFrame(columns=['R2', 'RMSECV'])
		kf = KFold(n_splits=self.k, shuffle=True, random_state=self.random_state)

		for train_i, test_i in kf.split(X):
			X_train, X_test = X.iloc[train_i], X.iloc[test_i]
			Y_train, Y_test = Y.iloc[train_i], Y.iloc[test_i]
			model = model.fit(X_train, Y_train)
			Y_pred = model.predict(X_test)
			r2 = r2_score(Y_test, Y_pred)
			rmse = math.sqrt(mean_squared_error(Y_test, Y_pred))
			metrics = metrics.append({'R2': r2, 'RMSECV': rmse}, ignore_index=True)

		metrics = metrics.append(metrics.mean(), ignore_index=True)
		indexes = list(metrics.index[:-1].values+1) + ['mean']
		metrics.index = indexes
		return metrics

	def pls_fitness_function(self, variables, X, Y):
		X_train, X_test, Y_train, Y_test = train_test_split(X[variables], Y, test_size=0.33, random_state=self.random_state)

		pls = self.pls.fit(X_train, Y_train)
		Y_pred = pls.predict(X_test)
		return explained_variance_score(Y_test, Y_pred)

	def rna_fitness_function(self, variables, X, Y):
		X_train, X_test, Y_train, Y_test = train_test_split(X[variables], Y, test_size=0.33, random_state=self.random_state)

		rna = self.rna.fit(X_train, Y_train)
		Y_pred = rna.predict(X_test)
		return math.sqrt(mean_squared_error(Y_test, Y_pred))

	def pls_rna_fitness_function(self, variables, X, Y):
		X_train, X_test, Y_train, Y_test = train_test_split(X[variables], Y, test_size=0.33, random_state=self.random_state)

		# train
		pls = self.pls.fit(X_train, Y_train)
		rna = self.rna.fit(pls.x_scores_, Y_train)

		# you gotta finish this bish
		pls = self.pls.fit(X_test)
		Y_pred = rna.predict(pls.x_scores)
		return math.sqrt(mean_squared_error(Y_test, Y_pred))

	def train(self, model, X_train, Y_train):
		model = model.fit(X_train, Y_train)
		Y_pred = model.predict(X_train)

		r2 = r2_score(Y_train, Y_pred)
		rmse = math.sqrt(mean_squared_error(Y_train, Y_pred))

		metrics = pd.Series(data=[r2, rmse], index=['R2', 'RMSEC'])
		return model, metrics

	# is this really necessary?
	def predict(self, model, X_test, Y_test):
		Y_pred = model.predict(X_test)

		r2 = r2_score(Y_test, Y_pred)
		rmse = math.sqrt(mean_squared_error(Y_test, Y_pred))

		metrics = pd.Series(data=[r2, rmse], index=['R2', 'RMSEP'])

		return metrics
