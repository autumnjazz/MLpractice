import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV

digits = load_digits()

dt_clf = DecisionTreeClassifier()
param_grid = {'max_depth' : [1,2,3,4,5,6,7,8,9,10], 'min_samples_split' : [2,3,4]}
kfold = KFold(n_splits = 5)
n_iter = 0
cv_accuracy = []

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

for train_index, test_index in kfold.split(X_train):
    X_fold_train, X_fold_vali = X_train[train_index], X_train[test_index]
    y_fold_train, y_fold_vali = y_train[train_index], y_train[test_index]
    
    grid_dtree = GridSearchCV(dt_clf, param_grid = param_grid, refit= True)
    grid_dtree.fit(X_fold_train, y_fold_train)
    
    fold_pred = grid_dtree.predict(X_fold_vali)
    
    n_iter += 1
    print('Iter : {0}, Accuracy : {1:.4f}'.format(n_iter ,accuracy_score(y_fold_vali, fold_pred)))
    
    cv_accuracy.append(accuracy_score(y_fold_vali, fold_pred))

cv_accuracy = sum(cv_accuracy)/len(cv_accuracy)
print("Mean Accuracy : {0:.4f}".format(cv_accuracy))

print('GridSearchCV best parameters : ', grid_dtree.best_params_)
print('GridSearchCV best accuracy : {0:.4f}'.format(grid_dtree.best_score_))
    
estimator = grid_dtree.best_estimator_
pred = estimator.predict(X_test)
test_accuracy = accuracy_score(y_test, pred)
print("predicted data accuracy : {0:.4f}".format(test_accuracy))




