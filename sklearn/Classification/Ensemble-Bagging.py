from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(bc.data, bc.target, test_size = 0.2, random_state = 121)

rfc = RandomForestClassifier()
param ={
    'n_estimators'      : [10],
    'max_depth'         : [1,2,3],
    'min_samples_leaf'  : [1,2],
    'min_samples_split' : [2,3]
}
grid_rfc = GridSearchCV(rfc, param_grid = param)
grid_rfc.fit(X_train, y_train)

print('best parameters  : ', grid_rfc.best_params_)
print('best accuracy    : {0:.4f}'.format(grid_rfc.best_score_))

estimator = grid_rfc.best_estimator_
pred = estimator.predict(X_test)
print('accuracy score   : {0:.4f}'.format(accuracy_score(y_test, pred)))

