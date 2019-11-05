import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.special import expit


def main():
    np.random.seed(0)
    n_samples = 100
    
    X = np.random.normal(size=n_samples)
    y = (X > 0).astype(np.float)
    X[X > 0] *= 5
    X += .7 * np.random.normal(size=n_samples)
    X = X[:, np.newaxis]

    C=1e5
    solver='lbfgs'
    clf = linear_model.LogisticRegression(C=C, solver=solver)
    clf.fit(X, y)
    
    plot_logistic_regression(clf, X, y)


def plot_logistic_regression(model, X_data, y_data):
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.scatter(X_data.ravel(), y_data, color='black', zorder=20)
    X_test = np.linspace(-5, 10, 300)

    loss = expit(X_test * model.coef_ + model.intercept_).ravel()
    plt.plot(X_test, loss, color='red', linewidth=3)

    ols = linear_model.LinearRegression()
    ols.fit(X_data, y_data)
    plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
    plt.axhline(.5, color='.5')

    plt.ylabel('y')
    plt.xlabel('X')
    plt.xticks(range(-5, 10))
    plt.yticks([0, 0.5, 1])
    plt.ylim(-.25, 1.25)
    plt.xlim(-4, 10)
    plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
               loc="lower right", fontsize='small')
    plt.tight_layout()
    # plt.show()
    # plt.savefig("Intro to ML/Logistic Regression.png")


if __name__ == "__main__":
    main()
