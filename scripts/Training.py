
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import math


class Training():

    def optimal_lv_num(self, X, Y, max_iter=100, scale=True):
        rmsecv = np.array([])
        for lv in np.arange(10):
            pls = PLSRegression(lv, max_iter, scale)
            rmsecv = np.append(rmsecv, math.sqrt(abs(cross_val_score(pls, X, Y, scoring='neg_mean_squared_error', cv=10).mean())))

        optimal = np.where(rmsecv==min(rmsecv))[0][0] + 1
        return optimal

    def pls(self, X, Y, lv_number, k, max_iter=100, scale=True, cross_validate):
        pls = PLSRegression(lv_number, max_iter, scale)

        if(cross_validate):
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
        else:
            rmsecv = math.sqrt(abs(cross_val_score(pls, X, Y, scoring='neg_mean_squared_error', cv=10).mean()))
            metrics = rmsecv

        return metrics

    def predict(self, model, X_test, Y_test):
        Y_pred = model.predict(X_test)

        r2 = r2_score(Y_test, Y_pred)
        mse = mean_squared_error(Y_test, Y_pred)

        return r2, mse
