import numpy as np
import matplotlib.pyplot as plt


def main():
    fig, axes = plt.subplots(2,2)
    
    # Scatter
    x = np.random.randn(50)
    y = np.random.randn(50)
    colors = np.random.randint(0,100, 50)
    sizes = 500*np.pi*np.random.rand(50)**2
    axes[0,0].scatter(x, y, c=colors, s=sizes, alpha=0.3)
    
    
    # Bar
    x = np.arange(10)
    axes[0,1].bar(x, x**2)
    
    
    # Multi-Bar
    x = np.random.rand(3)
    y = np.random.rand(3)
    z = np.random.rand(3)
    data =  [x, y, z]
    
    x_ax =  np.arange(3)
    for i in x_ax:
        axes[1,0].bar(x_ax, data[i], bottom=np.sum(data[:i], axis=0))
    axes[1,0].set_xticks(x_ax)
    axes[1,0].set_xticklabels(['A', 'B', 'C'])
    
    
    # Histogram
    data = np.random.randn(1000)
    axes[1,1].hist(data, bins=50)
    
    fig.savefig("Matplotlib2.png")
    plt.show()
    
if __name__ == "__main__":
    main()
