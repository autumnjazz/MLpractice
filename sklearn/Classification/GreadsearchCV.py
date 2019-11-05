import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def main():
    iris = load_iris()
    X_train, X_test, y_train,y_test = train_test_split(iris.data, iris.target)
    
    dtree = DecisionTreeClassifier()
    param_grid = {'max_depth' : [1,2,3], 'min_samples_split' : [2,3]}
    grid_dtree = GridSearchCV(dtree, param_grid = param_grid, cv = 10, refit= True, return_train_score=True)
    grid_dtree.fit(X_train, y_train)
    
    scores_df = pd.DataFrame(grid_dtree.cv_results_)
    scores_df[['params', 'mean_test_score', 'rank_test_score', 'split0_test_score', 'split1_test_score', 'split2_test_score']]
    
    print('GridSearchCV best parameters : ', grid_dtree.best_params_)
    print('GridSearchCV best accuracy : {0:.4f}'.format(grid_dtree.best_score_))
    
    estimator = grid_dtree.best_estimator_
    
    pred = estimator.predict(X_test)
    print('Test Dataset accuracy : {0:.4f}'.format(accuracy_score(y_test, pred)))
    
    
if __name__ == "__main__":
    main()
