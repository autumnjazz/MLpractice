import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

#K-Fold: Simple and Stratified

#Simple K-Fold Cross Validation
print("\n#Simple K-Fold Cross Validation")
iris = load_iris() 
dt_clf = DecisionTreeClassifier()

n_iter = 0
kfold = KFold(n_splits = 5)
cv_accuracy = []

for train_idx, test_idx in kfold.split(iris.data):
    n_iter += 1

    X_train, X_test = iris.data[train_idx], iris.data[test_idx]
    y_train, y_test = iris.target[train_idx], iris.target[test_idx]

    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("Iter : {0} Cross-Validation Accuracy : {1}, Train Data size : {2}, Test Data size : {3}"
          .format(n_iter, accuracy, train_size, test_size))

    cv_accuracy.append(accuracy)

print("Mean Accuracy : {0:.4f}%".format(np.mean(cv_accuracy)* 100))


#Stratified K-Fold Cross Validation
print("\n#Stratified K-Fold Cross Validation")

n_iter = 0
skf = StratifiedKFold(n_splits=3) # iris data has 3 classes
avg_acc = []

iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
iris_df['label'] = iris.target

for train_idx, test_idx in skf.split(iris.data, iris.target):
    n_iter += 1
    
    train_label = iris_df['label'].iloc[train_idx]                  
    test_label = iris_df['label'].iloc[test_idx]
    X_train, X_test = iris.data[train_idx], iris.data[test_idx]
    y_train, y_test = train_label, test_label
    
    print("Iteration :", n_iter)
    print("--------------------")
    print("train label distribution : \n", train_label.value_counts())
    print("--------------------")
    print("test label distribution : \n", test_label.value_counts())
    print("--------------------")
    
    dt_clf.fit(X_train, y_train)
    pred = dt_clf.predict(X_test)

    accuracy = np.round(accuracy_score(y_test, pred), 4)
    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    print("Iter : {0} Cross-Validation Accuracy : {1}, Train Data size : {2}, Test Data size : {3}"
          .format(n_iter, accuracy*100, train_size, test_size))

    avg_acc.append(accuracy)
    
print("Mean Accuracy : {0:.4f}%".format(np.mean(avg_acc)* 100))