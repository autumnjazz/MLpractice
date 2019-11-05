import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    x = np.arange(10)
    fig, ax = plt.subplots()
    
    ax.plot(
        x, x, label='y=x',
        marker='o',
        color='blue',
        linestyle=':'
    )
    ax.plot(
        x, x**2, label='y=x^2',
        marker='^',
        color='red',
        linestyle='--'
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    ax.set_xlim(0,10)
    ax.set_ylim(0,100)
    
    ax.legend(
        loc='upper left',
        shadow=True,
        fancybox=True,
        borderpad=2
    )
    
    plt.show()
    # fig.savefig("plot.png")

if __name__ == "__main__":
    main()
