import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
from evaluate import Evaluate

evaluator = Evaluate()


class CrossValidate():
    def optimal_value(self, err, values):
        index_of_best = err.index(min(err))
        if min(err) == err[0]:  #  if they're all the same
            index_of_best = 0
        return values[index_of_best]

    def do_cross_validation_kfold(self, X, y):
        k_folds = [2, 5, 10, 25, 50, 100]

        mean_mse = []
        var_mse = []
        std_mse = []
        for folds in k_folds:
            mse = []
            kf = KFold(n_splits=folds)
            print("KFold = ", folds)
            for train, test in kf.split(X):
                print("training...")
                model = LogisticRegression().fit(X[train], y[train])
                pred = model.predict(X[test])
                mse.append(mean_squared_error(y[test], pred))

            mean_mse.append(np.mean(mse))
            var_mse.append(np.var(mse))
            std_mse.append(np.std(mse))

        print("--- Results ---")
        print("-> KFold Cross Val. -> Recommended: Lowest variance @ KFolds =",
              optimal_value(mean_mse, k_folds))

        plt.figure()
        kf_vals = ['2', '5', '10', '25', '50', '100']
        plt.errorbar(k_folds,
                     mean_mse,
                     yerr=var_mse,
                     capsize=5,
                     ecolor='red',
                     label='Mean prediction error with varience')
        plt.title("KFold Cross Validation")
        plt.xlabel('K-folds')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def do_cross_validation_c(self, X, y):
        c_values = [0.01, 0.1, 1, 100]

        mean_mse = []
        var_mse = []
        std_mse = []
        for c in c_values:
            mse = []
            print("C = ", c)
            kf = KFold(n_splits=5)

            for train, test in kf.split(X):
                model = SVC(C=c, kernel='rbf').fit(X[train], y[train])
                pred = model.predict(X[test])
                mse.append(mean_squared_error(y[test], pred))

            mean_mse.append(np.mean(mse))
            var_mse.append(np.var(mse))
            std_mse.append(np.std(mse))

        print("--- Results ---")
        print(
            "-> Hyperparam C Cross Val. -> Recommended: Lowest variance @ C =",
            optimal_value(mean_mse, c_values))

        plt.figure()
        plt.errorbar(c_values,
                     mean_mse,
                     yerr=var_mse,
                     capsize=5,
                     ecolor='red',
                     label='Mean prediction error with varience')
        plt.title("Cross Validation for Hyperparameter C")
        plt.xlabel('Hyperparameter C')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

    def do_cross_validation_knn(self, X, y):
        knn_range = [50, 100, 150, 200, 350]

        mean_mse = []
        var_mse = []
        std_mse = []

        for KNN in knn_range:
            mse = []
            print("KNN = ", KNN)
            kf = KFold(n_splits=5)

            for train, test in kf.split(X):
                print("---")
                model = KNeighborsClassifier(n_neighbors=KNN).fit(
                    X[train], y[train])
                pred = model.predict(X[test])
                mse.append(mean_squared_error(y[test], pred))

            mean_mse.append(np.mean(mse))
            var_mse.append(np.var(mse))
            std_mse.append(np.std(mse))
        print("MEAN :", mean_mse)
        print("-> KNN Cross Val. -> Recommending: Lowest variance @ knn =",
              self.optimal_value(mean_mse, knn_range))

        plt.figure()
        knn_vals = ['50', '100', '150', '300', '350']

        plt.errorbar(knn_vals,
                     mean_mse,
                     yerr=var_mse,
                     capsize=5,
                     ecolor='red',
                     label='Mean prediction error with varience')
        plt.title("KNN Validation")
        plt.xlabel('KNN')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()