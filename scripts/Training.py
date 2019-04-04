
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


class Training():

    def pls(self, X, Y, lv_number, k, max_iter=100, scale=True):
        pls = PLSRegression(lv_number, max_iter, scale)
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        metrics = pd.DataFrame(columns=['R2', 'MSE'])

        for train_i, test_i in kf.split(X):
            X_train, X_test = X.iloc[train_i], X.iloc[test_i]
            Y_train, Y_test = Y.iloc[train_i], Y.iloc[test_i]

            pls.fit(X_train, Y_train)
            Y_pred = pls.predict(X_test)

            r2 = r2_score(Y_test, Y_pred)
            mse = mean_squared_error(Y_test, Y_pred)
            metrics.append({'R2': r2, 'MSE': mse}, ignore_index=True)

        metrics.set_index(metrics.index+1, inplace=True)
        return metrics

    def predict(self, model, X_test, Y_test):
        Y_pred = model.predict(X_test)

        r2 = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)

        return r2, mse
