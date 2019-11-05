import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()
X_train, X_test, y_train,y_test = train_test_split(cancer.data,cancer.target)

lr_clf = KNeighborsClassifier()
knn_clf = LogisticRegression()

vo_clf = VotingClassifier(estimators = [('lr', lr_clf),('knn',knn_clf)], voting = 'soft')
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)

print('Voting Classifier accuracy : {0:.4f}'.format(accuracy_score(y_test, pred)))

#Other classifiers
classifiers = [lr_clf, knn_clf]
for classifier in classifiers:
    classifier.fit(X_train, y_train)
    pred = classifier.predict(X_test)
    class_name = classifier.__class__.__name__
    print("{0} accuracy : {1:.4f}".format(class_name, accuracy_score(y_test, pred)))

