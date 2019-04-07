import numpy as np
import matplotlib.pyplot as plt

def display_data(x, num_rows, num_columns):
    
    x = np.concatenate([np.concatenate(x[(i*num_rows):((i+1)*num_rows)], 0) for i in range(num_columns)], 1)

    plt.imshow(x, cmap = 'gray')
    plt.axis('off')
    plt.show()

def plot_cost(errors):
    plt.plot(errors)
    plt.title("Cost for every iteration")
    plt.xlabel("iteration")
    plt.ylabel("Cost (J(theta))")
    plt.show()