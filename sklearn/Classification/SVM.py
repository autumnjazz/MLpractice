import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def main():
    np.random.seed(123)
    X = np.random.randn(300, 2)
    Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    
    ### kernel = 'rbf' 'poly' 'sigmoid' // 'linear'
    svm_kernel = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
    svm_kernel.fit(X,Y)

    plot_kernelSVM_result(svm_kernel, X, Y)


def plot_kernelSVM_result(svm_model, X, Y):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                         np.linspace(-3, 3, 500))

    # plot the decision function for each datapoint on the grid
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
               origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                           linestyles='dashed')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-3, 3, -3, 3])
    # plt.show()
    # plt.savefig("Intro to ML/SVM.png")
    
    
if __name__ == "__main__":
    main()