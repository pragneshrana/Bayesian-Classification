import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Fixing random state for reproducibility
np.random.seed(19680801)

N_points = 100000
n_bins = 20

# Generate a normal distribution, center at x=0 and y=5
x = [0.045,153,51,5.,16,65.1]

fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)

# We can set the number of bins with the `bins` kwarg
axs.hist(x, bins=n_bins)
plt.show()